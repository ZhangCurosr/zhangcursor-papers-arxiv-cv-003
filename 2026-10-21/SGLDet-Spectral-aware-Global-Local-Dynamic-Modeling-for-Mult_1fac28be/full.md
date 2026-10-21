# SGLDet: Spectral-aware Global-Local Dynamic Modeling for Multispectral Aerial Object Detection

Peng Zhang, Shuaihao Han, Tingfa Xu<sup>\*</sup>, and Jianan Li

Abstract—Multispectral aerial object detection aims to exploit spectral–spatial cues to address object detection challenges in complex aerial scenes. Existing methods mainly focus on spectral– spatial decoupling and object-aware spectral–spatial fusion to improve spectral utilization. However, two critical issues remain underexplored: 1) severe foreground–background imbalance caused by large scene coverage hinders reliable object recognition; 2) aerial objects exhibit substantial local variations in scale, orientation, and structure, which hinder accurate localization. To address these issues, we propose SGLDet, a spectral-aware global–local dynamic modeling framework for multispectral aerial object detection, which is motivated by the observation that object semantics are jointly determined by global spectral consistency and scene context, while precise localization requires dynamic feature modulation conditioned on the input spectral and spatial properties. To realize these properties, SGLDet consists of three coordinated components: a Spectral-driven Global Dynamic Modeling module that aggregates spectrally related features across the scene via similarity-guided sparse matching; a Multi-scale Global Context Integration module that constructs a compact scene memory for scene-guided contextual reasoning; and a Spectral-aware Input-dependent Calibration Aggregation module that performs expert-kernel aggregation with scene-guided spectral–spatial calibration to enable adaptive local feature modulation. Extensive experiments on MODA, HOD3K, and DroneVehicle demonstrate that SGLDet achieves superior performance with competitive computational cost. Code is available at https://github.com/Riiluo/SGLDet.

Index Terms—Multispectral aerial object detection, spectralaware modeling, global-local dynamic modeling.

## I. INTRODUCTION

ERIAL object detection is a fundamental task in remote sensing, with applications in urban management, traffic surveillance, maritime monitoring, and disaster response. Most existing methods rely on RGB inputs, whose limited spectral representation often struggles under illumination changes, oc clusions, and background interference, especially when objects resemble surrounding regions. Multispectral images (MSIs) provide richer spectral responses across multiple wavelengths and thus offer complementary cues beyond spatial appearance, making them a promising information source for aerial object detection. Existing methods for multispectral aerial object detection (MAOD) mainly either decouple spectral and spatial information for separate modeling or strengthen adaptive spectral-spatial fusion for target perception [1]–[3]. However, because these methods mainly focus on local spectral-spatial

![](images/e50046a019e2af0b25d750640066eb7ab3d31072f4a000ca23ecd44ad5d01a6b.jpg)  
Fig. 1. (a) Global spectral relations associate objects or regions with similar spectral responses across spatial locations, while global scene context further helps disambiguate visually confusing regions. SGDM and MGCI perform these two operations, respectively. (b) Local dynamic modulation produces more object-aligned responses and facilitates more accurate localization for objects in complex backgrounds. SICA implements this input-adaptive refinement.

interaction and emphasize feature enhancement within limited receptive fields, they remain inherently insufficient for resolving global semantic ambiguity in complex real-world aerial scenes.

Concretely, aerial images usually cover large spatial extents, and aerial objects often appear at small scales and are embedded in complex scenes. In MODA [2], for example, objects occupy only 0.15% of the image area on average, resulting in severe foreground–background imbalance, so local spectral– spatial cues alone are often insufficient for reliable recognition. Fortunately, MSI data provide a clue that is difficult to obtain from RGB imagery: objects composed of similar materials often exhibit consistent spectral signatures even across distant spatial locations, while scene context can further suppress visually confusing background regions. As shown in Figure 1(a), linking distant regions with similar spectral responses can reinforce objects with ambiguous local appearances while suppressing visually similar but spectrally distinct background responses. Scene context further resolves the remaining semantic ambiguity. These observations indicate that robust MAOD should integrate scene-wide spectral relations with scene-level context to better resolve semantic ambiguity in aerial scenes. Motivated by this observation, we introduce two complementary designs for global reasoning in multispectral aerial object detection. The first is a Spectral-driven Global Dynamic Modeling (SGDM) module, which retrieves spectrally similar responses through similarity-guided sparse matching to encourage semantically coherent representations for spectrally similar objects. The second is a Multi-scale Global Context Integration (MGCI) module, which consolidates cross-scale evidence into a compact

![](images/0b18196bd3e76f926638d8ab7801c4a4b3be43d34b152e023fb8e6d031d6c596.jpg)  
Fig. 2. Accuracy–complexity comparisons on MODA and HOD3K using mAP<sub>50</sub> (left) and mAP (right).

(b) HOD3K Training Scheme

scene memory and enhances scene-guided contextual reasoning through contextual retrieval.

However, resolving global semantic ambiguity is still insufficient for MAOD. Aerial objects often vary substantially in scale, orientation, and structure, whereas standard convolutions apply fixed kernels and receptive fields to all inputs. This mismatch makes it difficult to produce object-aligned responses in cluttered backgrounds, as shown in Figure 1(b). Local dynamic modulation therefore complements global reasoning by improving object localization. To address this issue, we further introduce a Spectral-aware Input-dependent Calibration Aggregation (SICA) module, which performs scene-dependent dynamic feature modulation through input-adaptive expert aggregation with spectral–spatial calibration for precise localization. Together, SGDM, MGCI, and SICA constitute SGLDet, a spectral-aware global-local dynamic modeling framework for multispectral aerial object detection.

Overall, SGLDet unifies global spectral dependency modeling, scene-guided contextual reasoning, and scene-dependent local feature modulation within a single framework. To evaluate the proposed method, we conduct extensive experiments on three benchmark datasets: MODA [2], HOD3K [1], and Dron eVehicle [4]. The results demonstrate that SGLDet achieves superior detection performance with competitive computational cost. As shown in Figure 2, SGLDet achieves stronger detection performance on both MODA and HOD3K with comparable computational cost. The main contributions are summarized as follows:

• We propose SGLDet, a spectral-aware global-local dynamic modeling framework for multispectral aerial object detection that unifies global spectral dependency modeling, scene-guided contextual refinement, and scene-dependent

dynamic local modulation.

• We develop a spectral-driven global reasoning design coupling similarity-guided sparse retrieval with sceneguided refinement, strengthening semantic consistency across spectrally similar objects and alleviating global semantic ambiguity in aerial scenes.

• We develop a spectral-aware input-dependent calibration aggregation mechanism with input-adaptive expert aggregation and spectral–spatial calibration for accurate localization under local variations.

## II. RELATED WORK

## A. Object Detection in Aerial Images

Object detection in RGB images has achieved remarkable progress in recent years, driven by advances in single-stage [5], two-stage [6], and Transformer-based detectors [7]. In aerial imagery, however, objects frequently appear with arbitrary orientations, necessitating specialized representations beyond horizontal bounding boxes. Two-stage approaches such as RoI Transformer [8] address rotation through region transformation, while single-stage oriented detectors such as R3Det [9], S<sup>2</sup>ANet [10], Oriented RepPoints [11], and CFA [12] improve oriented localization and dense object modeling. More recently, LSKNet [13] and Strip R-CNN [14] explore large-kernel designs to better capture broader spatial context in remote sensing scenes.

Beyond geometric modeling, recent aerial detectors further improve dense-instance perception through stronger spatial context modeling and feature aggregation [13]–[15]. However, these methods primarily rely on RGB spatial features and therefore remain limited when objects share similar visual appearances with cluttered backgrounds.

## B. Multispectral Object Detection

Leveraging information from continuous spectral bands, multispectral imagery holds significant potential for scene characterization and interpretation [16]. However, due to the limitations of multispectral image acquisition and resolution, datasets and methods for multispectral object detection remain relatively scarce. Multispectral data usually exhibit high corre lation and redundancy among adjacent bands, and early studies mainly focused on pixel-level analysis such as spectral matching or anomaly detection, rather than object-level localization.

Yan et al. [17] explored spectral image object detection at the object level. He et al. [1] proposed S2ADet and the HOD3K dataset, introducing a two-stream architecture that separately extracts spectral and spatial features before aggregating dual detection branches. Han et al. [2] further introduced MODA and proposed OSSDet to enhance target perception and suppress background interference through spectral-spatial interaction and object-aware cues. Xu et al. [3] subsequently proposed DAHNet, which decomposes multispectral data and aggregates complementary spectral-spatial representations through crossdomain synergistic attention. Despite these advances, spectral information is still mainly used as an auxiliary cue for local feature enhancement through dimensionality reduction or channel reweighting, rather than as a global semantic prior for unifying materially consistent objects across the scene [1]–[3].

## C. Spatial-Spectral Feature Learning

Spatial-spectral feature learning has been extensively studied in multispectral image analysis, where the primary objective is to jointly model spectral correlations and spatial context for pixel-level discrimination. Early deep learning methods applied 2D CNNs to extract spatial features from individual bands or reduced representations, while 3D CNNs were later introduced to capture inter-band dependencies through volumetric convolutions. Despite these advances, most spatialspectral learning paradigms are designed for classification or restoration rather than accurate instance-level localization in aerial detection.

When extended to object detection, spatial-spectral learning remains relatively underexplored. Existing methods usually treat spectral cues as auxiliary signals in feature fusion [1]–[3], so most strategies still operate at the local feature level without explicit modeling of global dependencies or adaptive local aggregation for object-specific variations across aerial scenes.

## III. METHOD

We present SGLDet, a spectral-aware global-local dynamic modeling framework for multispectral aerial object detection, as illustrated in Figure 3. SGLDet follows a global-to-local design to jointly address semantic ambiguity and localization challenges in complex aerial scenes. Specifically, the Spectral driven Global Dynamic Modeling (SGDM) module establishes scene-wide spectral dependencies through similarity-guided sparse retrieval, encouraging semantically consistent responses for spectrally similar objects. The Multi-scale Global Context Integration (MGCI) module then consolidates cross-scale evidence into a compact scene memory and performs sceneguided contextual retrieval for coherent global reasoning. Finally, the Spectral-aware Input-dependent Calibration Aggregation (SICA) module adaptively refines local features through input-dependent expert aggregation and scene-guided spectralspatial calibration, producing more object-aligned responses for accurate localization.

## A. Spectral driven Global Dynamic Modeling

In multispectral aerial imagery, objects with similar spectral signatures may appear at distant spatial locations, while visually similar regions can still exhibit distinct spectral responses. This property makes purely local interactions inadequate for capturing long-range spectral dependencies across the scene. To this end, we introduce a Spectral-driven Global Dynamic Modeling (SGDM) module, which performs sparse retrieval in the spectral domain using a negative squared Euclidean affinity score, followed by token refinement that promotes spectral consistency and incorporates contextual information.

Token projection and spectral affinity. Given an input feature $\pmb { F } \in \mathbb { R } ^ { \tilde { C } \times \tilde { H } \times W }$ , SGDM first transforms it into a token sequence and linearly projects it into query, key, and value embeddings:

$$
[ Q , K , V ] = [ \phi _ { q } ( { \pmb F } ) , \phi _ { k } ( { \pmb F } ) , \phi _ { v } ( { \pmb F } ) ] .\tag{1}
$$

where $Q , K , V \in \mathbb { R } ^ { N \times C }$ and N denotes the number of pooled spectral tokens. For multi-head modeling, we further partition them as $\pmb { Q } = [ \pmb { Q } ^ { ( 1 ) } , \ldots , \pmb { Q } ^ { ( H ) } ] , \pmb { K } = [ \pmb { K } ^ { ( 1 ) } , \ldots , \pmb { K } ^ { ( \hat { H } ) } ]$ , and $\pmb { V } = [ \pmb { V } ^ { ( 1 ) } , \dots , \pmb { V } ^ { ( H ) } ] ,$ , where H is the number of heads and each head lies in $\mathbb { R } ^ { N \times d }$ with $C = H d$ . To explicitly model global spectral dependencies, SGDM measures token affinity in each head using the negative squared Euclidean score:

$$
\begin{array} { r l } & { \Psi _ { i j } ^ { ( h ) } = \psi \Big ( Q _ { i } ^ { ( h ) } , \mathbf { { K } } _ { j } ^ { ( h ) } \Big ) } \\ & { \qquad = 2 \left( { { Q } } _ { i } ^ { ( h ) } \right) ^ { \top } \mathbf { { K } } _ { j } ^ { ( h ) } - \left\| \big . { { Q } } _ { i } ^ { ( h ) } \right\| _ { 2 } ^ { 2 } - \left\| \mathbf { { K } } _ { j } ^ { ( h ) } \right\| _ { 2 } ^ { 2 } , } \end{array}\tag{2}
$$

Equation (2) is a negative squared Euclidean affinity score, i.e., $\Psi _ { i j } ^ { ( h ) } = - \| \ b { Q } _ { i } ^ { ( h ) } - \check { \ b { K } } _ { j } ^ { ( h ) } \| _ { 2 } ^ { 2 }$ , so a larger score denotes a smaller distance and a more similar query–key pair. For brevity, let $\pmb q = \pmb Q _ { i } ^ { ( h ) }$ and $k = K _ { i } ^ { ( h ) }$ , and let $\theta _ { i j }$ denote the angle between them. Their squared distance can be decomposed as

$$
\begin{array} { r } { \| \pmb { q } - \pmb { k } \| _ { 2 } ^ { 2 } = \left( \| \pmb { q } \| _ { 2 } - \| \pmb { k } \| _ { 2 } \right) ^ { 2 } } \\ { + 2 \| \pmb { q } \| _ { 2 } \| \pmb { k } \| _ { 2 } \big ( 1 - \cos \theta _ { i j } \big ) . } \end{array}
$$

Hence, the negative squared Euclidean score accounts for both angular disagreement and norm mismatch, whereas cosine similarity and Spectral Angle Mapper (SAM) retain only angular information [18], [19]. This distinction is relevant because SGDM applies learned query and key projections without explicit $L _ { 2 }$ normalization after projection. Conversely, if both vectors were $L _ { \mathrm { { 2 } } \mathrm { { - n o r m a l i z e d } } } .$ , Eq. (2) would reduce to $2 \cos \theta _ { i j } - 2 ;$ because the negative SAM score $- \theta _ { i j } =$ − arccos(cos $\theta _ { i j } )$ is strictly increasing in cosine similarity, all three scores would then induce the same Top-K neighbor ordering. We therefore select the negative squared Euclidean score for the unnormalized SGDM feature space and validate this choice empirically in Section IV.

![](images/0191390f172e15c76a582211897380ad8a3cf0a56c34e44ae82b00d79be90b1c.jpg)  
Fig. 3. Overall architecture of SGLDet. The spectral-guided encoder first applies SGDM to establish long-range spectral dependencies through sparse retrieval of spectrally similar responses. The scene-conditioned detection decoder then couples MGCI and SICA for hierarchical refinement. (a) SGDM performs sparse spectral retrieval followed by spectral consensus formation and contextual integration. (b) MGCI consolidates distributed cross-scale scene evidence into a compact scene memory and reads out adaptive scene context, while SICA performs scene-dependent adaptive local filtering before detection.

Top-K sparse spectral retrieval. To avoid redundant global interaction, we retain only the Top-K most relevant neighbors for each query token, with $K = \lfloor r N \rfloor$ . After masking out all but the Top-K entries in $\Psi _ { i , : } ^ { ( h ) }$ , we set $\widetilde { \Psi } _ { i , j } ^ { ( h ) } ~ = ~ \Psi _ { i , j } ^ { ( h ) }$ for retained entries and −∞ otherwise. The resulting sparse retrieval distribution is then given by

$$
\Pi _ { i , j } ^ { ( h ) } = \frac { \exp \bigl ( \widetilde { \Psi } _ { i , j } ^ { ( h ) } \bigr ) } { \sum _ { t = 1 } ^ { N } \exp \bigl ( \widetilde { \Psi } _ { i , t } ^ { ( h ) } \bigr ) } , \quad j = 1 , \ldots , N .\tag{3}
$$

The resulting vector $\Pi _ { i . : } ^ { ( h ) }$ lies on the (N − 1)-dimensional probability simplex $\begin{array} { r } { \Delta ^ { N - 1 } = \{ \pmb { p } \in \mathbb { R } ^ { N } \ | \ p _ { j } \geq 0 , \ \sum _ { i = 1 } ^ { N } p _ { j } = } \end{array}$ 1}. Because the Top-K mask assigns zero probability to all unselected neighbors, $\Pi _ { i , : } ^ { ( h ) }$ is a K-sparse point on this simplex. This distribution corresponds to the Top-K spectral retrieval stage in Fig. 3(a), where only the most spectrally compatible neighbors are preserved for subsequent aggregation. After neighbor selection, sparse value aggregation retains NK query– key pairs and scales as

$$
\sum _ { i = 1 } ^ { N } K = \mathcal { O } ( N K ) , \qquad K \ll N .
$$

Spectral context aggregation. Unlike conventional sparse attention, which uses query–key affinity mainly to weight values, SGDM also aggregates selected keys into a spectral context that refines the query before output mapping. Non-local similarity is also used in hyperspectral reconstruction [20]; in SGDM, learned affinity selects features for object detection. Specifically, the gathered key and value entries corresponding to the Top-K neighbors are aggregated to form a spectral context vector for each query:

$$
\begin{array} { r } { { \pmb { c } } _ { i } ^ { ( h ) } = \left( { \pmb { K } } ^ { ( h ) } \right) ^ { \top } { \pmb { \Pi } } _ { i } ^ { ( h ) } + \beta \left( { \pmb { V } } ^ { ( h ) } \right) ^ { \top } { \pmb { \Pi } } _ { i } ^ { ( h ) } , } \end{array}\tag{4}
$$

where $\beta$ is a learnable coefficient that balances spectral consensus from keys and contextual information from values.

The spectral context $\pmb { c } _ { i } ^ { ( h ) }$ is then used in two complementary paths. First, it is added to the original query to obtain the spectrally refined query $\bar { \pmb q } _ { i } ^ { ( h ) } = { \pmb Q } _ { i } ^ { ( h ) } + { \pmb c } _ { i } ^ { ( h ) }$ , which aligns each query with its spectrally nearest neighborhood, strengthening global spectral consistency while preserving token identity. Second, the spectral context is passed through a mapping module $\phi _ { \mathrm { m } } ( \cdot )$ and then integrated with the refined query via residual-scaled injection followed by nonlinear refinement:

$$
{ \pmb { o } } _ { i } ^ { ( h ) } = \mathcal { F } \Big ( \bar { \pmb { q } } _ { i } ^ { ( h ) } + \gamma \phi _ { \mathrm { m } } ( { \pmb { c } } _ { i } ^ { ( h ) } ) \Big ) ,\tag{5}
$$

where $\phi _ { \mathrm { m } } ( \cdot )$ is a linear projection that transforms the spectral context, γ is a learnable scaling coefficient initialized near zero for training stability, and $\mathcal F ( \cdot )$ denotes a nonlinear feature refinement module that enhances the fused representation. Finally, the responses across all heads are concatenated and projected back to the original embedding space:

$$
\begin{array} { r } { \hat { \pmb { F } } _ { i } = \phi _ { o } \Big ( \mathrm { C o n c a t } \left[ \pmb { o } _ { i } ^ { ( 1 ) } , \dots , \pmb { o } _ { i } ^ { ( H ) } \right] \Big ) , } \end{array}\tag{6}
$$

where $\phi _ { o } ( \cdot )$ denotes the output projection. By routing the spectral context through both the query refinement path and the mapping module before integration, SGDM ensures that the final representation jointly captures spectral coherence and contextual evidence.

Overall, SGDM can be interpreted as a global spectral dependency modeling module for semantic unification rather than a standard dense attention operator. By jointly aggregating key and value information into a unified spectral context and routing it through both query refinement and the final mapping module, SGDM strengthens long-range spectrally guided interaction while preserving token identity for subsequent contextual reasoning. In practice, for the ResNet backbone used in our detector, we instantiate SGDM on the last three backbone stages rather than on the earliest low-level stage, where spectral semantics are less stable and global spectral retrieval is less beneficial.

![](images/8d6e20fd2825e3478ca644febd3ddc03c790ef93d5389dc1a0bf3e81798a7ce8.jpg)  
Fig. 4. Illustration of the proposed SICA module. SICA performs inputdependent local expert aggregation on $\boldsymbol { X } _ { l }$ and scene-guided calibration from $\bar { C _ { l } }$ to adaptively refine multilevel local features.

## B. Multi-scale Global Context Integration

While SGDM establishes spectrally guided long-range dependencies within each pyramid level, reliable recognition in multispectral aerial imagery further requires stable scenelevel contextual reasoning, especially when tiny foreground objects are immersed in cluttered backgrounds under severe foreground-background imbalance. The responses refined by SGDM retain their pyramid levels and spatial identities and therefore remain distributed across scales rather than forming a coherent scene representation. We therefore introduce a Multiscale Global Context Integration (MGCI) module, which aligns and consolidates this evidence into a compact scene memory and lets each spatial position retrieve an adaptive context for subsequent local refinement.

Cross-scale scene prior construction. Let $\{ F _ { l } \} _ { l = 1 } ^ { 4 }$ denote the hierarchy refined by SGDM. MGCI is built upon an intermediate feature scale that balances semantic abstraction and spatial detail. At this scale, a scene context $S _ { \mathrm { p } } \in \mathbb { R } ^ { C \times H _ { 2 } \times W _ { 2 } }$ collects higher-level SGDM evidence, where $H _ { 2 }$ and $W _ { 2 }$ denote the spatial size aligned with $\pmb { F } _ { 2 } ^ { \prime }$

$$
S _ { \mathrm { p } } = f _ { \mathrm { p } } \big ( \big [ \phi _ { 2 } ( { \pmb F } _ { 2 } ) , \mathcal { P } _ { s } \big ( { \pmb F } _ { 3 } + \phi _ { 4 } ( \mathcal { P } _ { s } ( { \pmb F } _ { 4 } ) ) \big ) \big ] \big ) .\tag{7}
$$

Meanwhile, the scene content $S _ { \mathrm { c } } ~ \in ~ \mathbb { R } ^ { C \times H _ { 2 } \times W _ { 2 } }$ preserves spatial detail through $S _ { \mathrm { c } } = f _ { \mathrm { c } } ( [ \tilde { \cal F } _ { 2 } , { \cal D } ( \tilde { \cal F } _ { 1 } ) ] )$ , where $\dot { \pmb { F } } _ { 1 }$ and ${ \tilde { \pmb F } } _ { 2 }$ denote the corresponding features after channel alignment. Here, $\mathcal { P } _ { s } ( \cdot )$ denotes pixel shuffle, $\mathcal { D } ( \cdot )$ denotes stride-2 downsampling, and $f _ { \mathrm { p } } ( \cdot )$ and $f _ { \mathrm { c } } ( \cdot )$ are fusion mappings.

Role separation between context and content. The context branch summarizes cross-scale scene evidence into a scene context, whereas the content branch preserves the finer spatial detail used to query the compact scene memory. This complementary role separation ensures that MGCI does not simply repeat the spectral retrieval behavior of SGDM, but instead reorganizes distributed scene evidence into a global scene representation for subsequent decoding.

Scene token formation and memory initialization. Subsequently, MGCI writes distributed scene evidence into compact scene memory and then reads an adaptive scene context from this memory, so scene-level reasoning operates on compact scene memory rather than dense token interactions. We first project the scene context as $\bar { \pmb { S } } _ { \mathrm { p } } = \phi _ { \mathrm { p } } ( \pmb { S } _ { \mathrm { p } } )$ and then construct scene tokens by coupling the scene content with this projected scene context and their discrepancy at the same decoding scale:

$$
\pmb { T } = \mathcal { R } \big ( f _ { \mathrm { s } } \big ( [ S _ { \mathrm { c } } , \bar { S } _ { \mathrm { p } } , | S _ { \mathrm { c } } - \bar { S } _ { \mathrm { p } } | ] \big ) \big ) \in \mathbb { R } ^ { N \times C } .\tag{8}
$$

where $\mathcal { R } ( \cdot )$ denotes adaptive pooling followed by tokenization and $N = H _ { s } W _ { s }$ . The three-way concatenation couples local spatial detail $( S _ { \mathrm { c } } )$ with the projected scene context $( \bar { \cal S } _ { \mathrm { p } } )$ and their absolute deviation, so that scene tokens encode both global context and local deviations from it. The M per-image scene slots are initialized by $\pmb { { \cal E } } = \pmb { { \cal E } } _ { 0 } + \pmb { { \cal E } } _ { \mathrm { g } } \in \mathbb { R } ^ { M \times C }$ , where $\scriptstyle { E _ { 0 } }$ denotes learnable memory bases and $E _ { \mathrm { g } }$ denotes group embeddings. These learned bases initialize a compact scene memory, which is conditioned on the current input through the write operation and used for subsequent context retrieval.

Memory writing. We use memory queries together with scene token keys and values, defined as $Q _ { \mathrm { w } } ~ = ~ \phi _ { \mathrm { w } } ^ { q } ( E )$ $\begin{array} { r l } { K _ { \mathrm { w } } } & { { } = } \end{array}$ $\phi _ { \mathrm { w } } ^ { k } ( \pmb { T } )$ , and $V _ { \mathrm { w } } = \phi _ { \mathrm { w } } ^ { v } ( T )$ . The write stage becomes

$$
M = \mathrm { L N } ( E + \mathrm { M H C A } _ { \mathrm { w } } ( Q _ { \mathrm { w } } , K _ { \mathrm { w } } , V _ { \mathrm { w } } ) ) \in \mathbb { R } ^ { M \times C } ,\tag{9}
$$

where MHCA $( Q , K , V ) = \mathrm { S o f t m a x } ( Q K ^ { \top } / \sqrt { d } ) V$ denotes multi-head cross-attention for slot writing and reading, with $\star \in \{ \mathrm { w } , \mathrm { r } \}$ . The write stage forms M input-conditioned scene slots from N scene tokens and contains $\mathcal { O } ( M N )$ query–key score pairs.

Memory reading and contextual refinement. For memory reading, scene content queries the compact scene memory to obtain a position adaptive scene context. We instantiate this read step with $Q _ { \mathrm { r } } = \phi _ { \mathrm { r } } ^ { q } ( S _ { \mathrm { c } } ) , K _ { \mathrm { r } } = \phi _ { \mathrm { r } } ^ { k } ( M )$ , and $V _ { \mathrm { r } } = \phi _ { \mathrm { r } } ^ { v } ( M )$ The read stage can be formulated as

$$
S _ { \mathrm { r e f } } = S _ { \mathrm { c } } + \gamma \phi _ { \mathrm { o } } \big ( \mathrm { M H C A } _ { \mathrm { r } } ( Q _ { \mathrm { r } } , K _ { \mathrm { r } } , V _ { \mathrm { r } } ) \big ) .\tag{10}
$$

where $\hat { \pmb { S } } = \phi _ { 0 } ( \mathrm { M H C A _ { r } } ( \pmb { Q } _ { \mathrm { r } } , \pmb { K } _ { \mathrm { r } } , \pmb { V } _ { \mathrm { r } } ) ) \in \mathbb { R } ^ { C \times H _ { 2 } \times W _ { 2 } }$ denotes the retrieved scene context and $\gamma$ is a learnable residual coefficient. Rather than uniformly broadcasting a single global context, this scene-conditioned readout lets each position query the per-image scene slots. Let $P = H _ { 2 } W _ { 2 }$ denote the number of scene-content queries. The read stage contains $\mathcal { O } ( P M )$ query–key score pairs, so slot writing and reading together require $\mathcal { O } ( M N + P M ) = \mathcal { O } ( M ( N + P ) )$ , compared with $\mathcal { O } ( P N )$ for direct MHCA from $P$ queries to N scene tokens. With fixed M, this compact representation reduces cross-scale interaction cost while preserving position-specific retrieval from a shared image-conditioned scene state.

## C. Spectral-aware Input-dependent Calibration Aggregation

While SGDM and MGCI establish global spectral consistency and scene-level contextual coherence, accurate aerial object detection still depends on how local neighborhoods are aggregated. Aerial objects exhibit substantial variation in scale, orientation, aspect ratio, and surrounding clutter, whereas standard convolution applies an identical aggregation pattern to all spatial locations. Such static local modeling is often insufficient for multispectral aerial imagery, where the informative spectral–spatial structure can change significantly across scenes and object instances. To address this issue, we introduce a Spectral-aware Input-dependent Calibration Aggregation (SICA) module, which couples input-dependent local expert aggregation from $X _ { l }$ with scene-guided calibration from $C _ { l }$ for adaptive local refinement.

TABLE I  
COMPARISON WITH OTHER METHODS ON MODA. <sup>†</sup> INDICATES METHODS ORIGINALLY DESIGNED FOR MULTISPECTRAL OBJECT DETECTION.  
```python
Algorithm 1 Pseudo-code of hierarchical SICA refinement
# X_pyr: local pyramid features at levels {2, 3, 4}
# S_ref: 256-channel MGCI context at level 2
# W: depth-wise expert bank with M experts
# Gamma[3:4]: separate 3x3 stride-2 convolutions
# (256->256 channels; padding 1; BN + ReLU)
def hierarchical_sica(X_pyr, S_ref, W, Gamma):
refined = {}
C = S_ref # Gamma[2] = Identity
for level in [2, 3, 4]:
x_desc = GAP(X_pyr[level])
alpha = Softmax(route_mlp(x_desc))
W_mix = sum(alpha[m] <sub>*</sub> W[m] for m in range(M))
Z = AdaptiveFiltering(X_pyr[level], W_mix)
gate_c = Sigmoid(channel_mlp(GAP(C)))
gate_s = Sigmoid(spatial_conv(cat(AvgPool(C),
MaxPool(C))))
F = out_proj(Z <sub>*</sub> gate_c + C <sub>*</sub> gate_s)
refined[level] = F
if level < 4:
C = Gamma[level + 1](F)
return refined
```

<table><tr><td>Method</td><td>car</td><td>bus</td><td>van</td><td>aw-bike</td><td>truck</td><td>tricycle</td><td></td><td>bike pedestrian</td><td> $\bf { m A P } _ { 5 0 }$  ←</td><td> $\mathbf { m A P _ { 7 5 } } \uparrow$ </td><td>mAP↑</td><td>FLOPs↓</td><td>Params↓</td></tr><tr><td colspan="10">two-stage methods</td><td colspan="5"></td></tr><tr><td>Roi Transformer [8]</td><td>90.5</td><td></td><td>89.3 75.2</td><td>73.3</td><td>68.7</td><td>51.8</td><td>44.5</td><td>30.0</td><td>65.4 65.5</td><td>43.4</td><td>40.7</td><td></td><td>244.7G</td><td>55.3M</td></tr><tr><td>O-RCNN [21]</td><td>90.5</td><td>89.7</td><td>74.6</td><td>72.5</td><td>66.6</td><td>54.7</td><td>45.1</td><td>30.4</td><td></td><td></td><td>44.0</td><td>40.9</td><td>244.7G</td><td>55.3M</td></tr><tr><td>LSKNet [13]</td><td>90.4</td><td>88.5</td><td>74.5</td><td>72.5</td><td>66.0</td><td>52.5</td><td>41.0</td><td>28.2</td><td>64.2</td><td></td><td>42.8</td><td>39.4</td><td>221.9G</td><td>32.6M</td></tr><tr><td>StripRCNN [14]</td><td>90.5</td><td>89.0</td><td>76.1</td><td>73.6</td><td>67.1</td><td>56.5</td><td>45.2</td><td>30.7</td><td>66.1</td><td></td><td>44.0</td><td>41.0</td><td>231.8G</td><td>45.2M</td></tr><tr><td>ReDet [22]</td><td>90.6</td><td>89.6</td><td>76.8</td><td>72.9</td><td>69.6</td><td>52.4</td><td>41.5</td><td>41.0</td><td>66.8</td><td></td><td>44.1</td><td>41.3</td><td>245.2G</td><td>31.6M</td></tr><tr><td colspan="10">one-stage methods</td><td colspan="4"></td></tr><tr><td>Rot. Retina [23]</td><td>90.2</td><td></td><td>81.972.4</td><td>59.0</td><td>45.5</td><td>30.8</td><td>21.7</td><td>25.4</td><td></td><td>53.4</td><td>37.3</td><td>34.1</td><td>233.5G</td><td>36.5M</td></tr><tr><td>R3Det [9]</td><td>90.4</td><td>88.7</td><td>74.0</td><td>63.8</td><td>57.8</td><td>45.1</td><td>29.4</td><td>32.2</td><td>60.2</td><td></td><td>36.7</td><td>34.8</td><td>362.2G</td><td>42.0M</td></tr><tr><td>S2 ANet [10]</td><td>90.4</td><td>88.7</td><td>74.4</td><td>69.1</td><td>62.7</td><td>48.9</td><td>30.1</td><td>40.7</td><td></td><td>63.1</td><td>38.7</td><td>36.7</td><td>216.5G</td><td>38.8M</td></tr><tr><td>S2ADet† [1]</td><td>90.3</td><td>86.6</td><td>72.1</td><td>71.3</td><td>57.2</td><td>54.1</td><td>35.0</td><td>40.8</td><td></td><td>63.5</td><td>41.1</td><td>38.9</td><td>406.0G</td><td>65.2M</td></tr><tr><td>Rot. FCOS [5]</td><td>90.3</td><td>86.3</td><td>75.2</td><td>71.2</td><td>59.0</td><td>53.8</td><td>34.7</td><td>37.4</td><td></td><td>63.5</td><td>41.8</td><td>39.1</td><td>226.9G</td><td>32.2M</td></tr><tr><td>GWD [24]</td><td>90.4</td><td>79.9</td><td>70.7</td><td>59.7</td><td>49.3</td><td>30.8</td><td>26.7</td><td>29.7</td><td></td><td>54.7</td><td>37.8</td><td>34.6</td><td>233.5G</td><td>36.5M</td></tr><tr><td>RepPts-R [25]</td><td>89.9</td><td>70.2</td><td>67.0</td><td>66.1</td><td>50.2</td><td>41.5</td><td>33.3</td><td>40.5</td><td></td><td>57.3</td><td>34.9</td><td>33.0</td><td>213.6G</td><td>36.9M</td></tr><tr><td>R3Det-KLD [26] CFA [12]</td><td>90.4</td><td>88.7</td><td>74.8</td><td>66.6</td><td>60.9</td><td>42.6</td><td>29.8</td><td>35.5</td><td></td><td>61.2</td><td>39.4</td><td>36.8</td><td>306.7G</td><td>39.6M</td></tr><tr><td>Ori. RepPoints [11]</td><td>90.4 90.5</td><td>89.0</td><td>76.7</td><td>69.7 71.2</td><td>64.4 66.2</td><td>55.1 53.1</td><td>43.1</td><td>41.5</td><td></td><td>66.2</td><td>43.2</td><td>40.6</td><td>213.6G</td><td>36.9M</td></tr><tr><td>RTMDet [27]</td><td>90.0</td><td>89.2</td><td>77.7</td><td>71.1</td><td>62.8</td><td>52.5</td><td>43.0 42.9</td><td>41.1</td><td></td><td>66.5</td><td>44.1</td><td>40.9</td><td>213.6G</td><td>36.9M</td></tr><tr><td></td><td>90.6</td><td></td><td>88.4 75.3</td><td>71.3</td><td>67.2</td><td>55.5</td><td></td><td>41.0</td><td></td><td>65.5</td><td>41.5</td><td>40.1</td><td>258.5G</td><td>52.3M</td></tr><tr><td>DCFL [28]</td><td></td><td>89.7 77.5</td><td></td><td></td><td></td><td></td><td>44.2</td><td>44.0</td><td></td><td>67.5</td><td>44.8</td><td>41.5</td><td>241.0G</td><td>37.0M</td></tr><tr><td>MessDet [15] OSSDet† [2]</td><td>90.0</td><td>89.1</td><td>75.4 79.2</td><td>71.4 72.7</td><td>70.2 69.7</td><td>54.9 58.8</td><td>48.4 45.3</td><td>45.5</td><td></td><td>68.1</td><td>42.4</td><td>40.1</td><td>616.0G</td><td>15.1M</td></tr><tr><td></td><td>90.5</td><td>89.9</td><td></td><td></td><td></td><td></td><td></td><td>45.7</td><td></td><td>69.0</td><td>45.9</td><td>42.7</td><td>263.1G</td><td>36.5M</td></tr><tr><td colspan="10">DETR-based methods</td><td colspan="5"></td></tr><tr><td>AO2-DETR [29] 89.4 88.1 76.0</td><td></td><td></td><td></td><td>69.5</td><td>65.7</td><td>54.8</td><td>41.9</td><td>41.0</td><td>65.8</td><td>43.0</td><td>40.0</td><td></td><td>240.0G</td><td>42.0M</td></tr><tr><td>ARS-DETR [30]</td><td>90.5</td><td>89.2</td><td>77.2</td><td>70.6</td><td>64.8</td><td>53.6</td><td>43.2</td><td>39.7</td><td>66.1</td><td></td><td>43.4</td><td>40.5</td><td>254.0G</td><td>44.2M</td></tr><tr><td>RHINO [31]</td><td>90.7</td><td>89.1</td><td>77.3</td><td>70.8</td><td>65.6</td><td>54.0</td><td>42.9</td><td>41.0</td><td>66.4</td><td></td><td>44.0</td><td>40.8</td><td>319.5G</td><td>50.8M</td></tr><tr><td>Ori-DETR [32]</td><td>90.8</td><td>89.3</td><td>78.8</td><td>72.8</td><td>68.0</td><td>56.0</td><td>46.6</td><td>40.1</td><td></td><td>67.8</td><td>45.1</td><td>41.9</td><td>492.0G</td><td>57.6M</td></tr><tr><td>SGLDet (Ours)</td><td></td><td>|90.5 90.2 79.3</td><td></td><td>75.8</td><td>70.8</td><td>65.6</td><td>51.5</td><td>50.9</td><td></td><td>71.8</td><td>47.0</td><td>44.3</td><td>241.6G</td><td>37.1M</td></tr></table>

As illustrated in Fig. 4, SICA decomposes local refinement into an input-dependent expert aggregation branch and a sceneguided calibration branch. The former produces adaptive local responses from $X _ { l } ,$ while the latter injects scene priors from $C _ { l }$ to modulate the routed response.

Input-dependent expert routing. Given a local feature map $\pmb { X } _ { l } \in \mathbb { R } ^ { C \times H \times W }$ and its scene guidance $C _ { l } \ \in \ \mathbb { R } ^ { C \times H \times W }$ SICA summarizes local content via global average pooling,

$$
\bar { \pmb { x } } _ { l } = \mathrm { G A P } ( \pmb { X } _ { l } ) .\tag{11}
$$

Based on this descriptor, SICA predicts routing coefficients over an expert bank,

$$
\alpha _ { l , m } = \frac { \exp ( \rho _ { m } ( \eta _ { \mathrm { r } } ( \bar { x } _ { l } ) ) ) } { \sum _ { t = 1 } ^ { M } \exp ( \rho _ { t } ( \eta _ { \mathrm { r } } ( \bar { x } _ { l } ) ) ) } , \qquad m = 1 , \ldots , M ,\tag{12}
$$

where M denotes the number of learnable depth-wise experts. Let $\{ W _ { m } \} _ { m = 1 } ^ { M }$ be the expert kernel bank with $W _ { m } \in \mathbf { \Sigma }$ $\mathbb { R } ^ { C \times \tilde { 1 } \times k \times k } ,$ . The routed kernel defines an adaptive filtering operator on $X _ { l } .$ , and the corresponding adaptive local response is written as

$$
\widetilde { \pmb { W } } _ { l } = \sum _ { m = 1 } ^ { M } \alpha _ { l , m } \pmb { W } _ { m } , \qquad \widetilde { \pmb { Z } } _ { l } = \mathrm { D w C o n v } ( \pmb { X } _ { l } ; \widetilde { \pmb { W } } _ { l } ) .\tag{13}
$$

This formulation preserves the structural prior of convolution while allowing the local aggregation pattern to be adaptively selected from a compact expert bank.

Scene-guided spectral-spatial calibration. In parallel, the scene guidance $C _ { l }$ is used to calibrate the routed response in

complementary channel and spatial manners. The channel gate is estimated from global scene statistics,

$$
\pmb { g } _ { l } ^ { \mathrm { c } } = \sigma \big ( \psi _ { \mathrm { c } } \big ( \mathrm { G A P } ( \pmb { C } _ { l } ) \big ) \big ) ,\tag{14}
$$

while the spatial gate is predicted from channel-wise pooled scene cues,

$$
\begin{array} { r } { G _ { l } ^ { \mathrm { s } } = \sigma \big ( \psi _ { \mathrm { s } } \big ( [ \mathrm { A v g } ( C _ { l } ) ; \mathrm { M a x } ( C _ { l } ) ] \big ) \big ) . } \end{array}\tag{15}
$$

where $\mathrm { A v g ( \cdot ) }$ and Max(·) denote channel-wise average and max pooling. The final refined response is obtained by projecting the scene-calibrated local response,

$$
\begin{array} { r } { \pmb { F } _ { l } ^ { \prime } = \phi _ { \mathrm { o } } \Big ( \widetilde { \pmb { Z } } _ { l } \odot \pmb { g } _ { l } ^ { \mathrm { c } } + \pmb { G } _ { l } ^ { \mathrm { s } } \odot \pmb { C } _ { l } \Big ) , } \end{array}\tag{16}
$$

where $\phi _ { \mathrm { o } } ( \cdot )$ denotes the output projection. Algorithm 1 summarizes the hierarchy. Here, $\Gamma _ { 2 } = \mathrm { I d }$ , whereas $\Gamma _ { 3 }$ and $\Gamma _ { 4 }$ are independent $3 \times 3$ convolutions with stride 2 and padding 1, followed by batch normalization and ReLU. In practice, SICA is applied from shallow to deep levels, passing scene-aware refinement to higher-level features. Consequently, its adaptive local filtering complements the global reasoning of SGDM and MGCI with content-aware refinement.

## IV. EXPERIMENTS

We evaluate SGLDet on three benchmarks with different input configurations and box annotations, including multiband images that retain wavelength-resolved information [33], [34]. MODA [2] uses eight-band images with oriented boxes; HOD3K [1] uses 16-band images with horizontal boxes; and DroneVehicle [4] provides paired RGB–infrared images with oriented boxes. The MODA and HOD3K comparisons therefore follow oriented- and horizontal-box evaluation protocols, respectively. Within these settings, MODA covers two-stage, one-stage, and DETR-based oriented detectors alongside multispectral methods; HOD3K covers horizontalbox detectors; and DroneVehicle includes single-modality and RGB–infrared fusion methods. We report $\mathrm { \ m A P _ { 5 0 } }$ and mAP on all three benchmarks. For MODA and HOD3K, we additionally report $\mathrm { \ m A P _ { 7 5 } }$ . We also report FLOPs, model parameters, and hardware-controlled wall-clock measurements. Comparison results are taken from prior studies or obtained in this work. Component ablations and detector compatibility studies are conducted on MODA; the inputmodality and fusion comparison is conducted on DroneVehicle.

Unless otherwise specified, all SGLDet training experiments and ablation variants were implemented in MMRotate with Py Torch 1.11.0 and conducted on two NVIDIA RTX 3090 GPUs (48 GB total). SGLDet builds on Oriented RepPoints [11] while preserving the original detection head. ResNet-50 pretrained on ImageNet was adopted as the backbone, and AdamW was used with a weight decay of 0.03 and gradient clipping with a maximum norm of 25. For each benchmark, a separate model was trained using its native multichannel input. Only the first ResNet-50 convolution was randomly initialized; the remaining backbone layers inherited ImageNet-pretrained weights. The ResNet-50 input stem is adapted to each dataset’s channel count while preserving its output width. The three-module configuration (SGDM+MGCI+SICA) follows the same featureprocessing flow across the three input settings.

## A. Results on the MODA Dataset

Data and Setups. The MODA dataset [2] is a large-scale multispectral aerial benchmark with 14,041 images and 330,191 oriented annotations over 8 categories. Each image has a resolution of $1 2 0 0 \times 9 0 0$ and covers 8 spectral bands from approximately 395 to 950 nm. MODA contains diverse urban scenes with small objects, low illumination, occlusion, truncation, and cluttered backgrounds. Each image was normalized and zero-padded from $1 2 0 0 \times 9 0 0$ to $1 2 1 6 \times 9 2 8$ . The model was trained for 12 epochs with a total batch size of 4. The learning rate was initialized to $7 \times 1 0 ^ { - 5 }$ and scheduled by cosine annealing with 700 linear warm-up iterations, a warmup ratio of $1 / 3 ,$ , and a minimum learning rate of $7 \times 1 0 ^ { - 7 }$

Quantitative Results. Table I lists the quantitative results of state-of-the-art detectors and our methods on the MODA dataset. SGLDet achieves the best overall detection performance and outperforms the strongest MSI-oriented baseline, OSSDet, by 2.8%, 1.1%, and 1.6% in $\mathrm { m A P _ { 5 0 } , \ m A P _ { 7 5 } }$ , and mAP, respectively. In class-wise evaluations, SGLDet achieves the best results in seven out of eight categories, with the largest gain of 6.8% on “tricycle”, showing the effectiveness of spectral-cueguided global and local dynamic modeling for multispectral aerial object detection.

Performance on Challenging Subsets. To construct three fixed challenging subsets, we compute three model-independent scores for each test image i:

$$
\begin{array} { l } { \displaystyle { L _ { i } = \frac { 1 } { 8 } \sum _ { c = 1 } ^ { 8 } \left. X _ { i , c } \right. _ { p } , } } \\ { \displaystyle { S _ { i } = \operatorname* { m e d i a n } _ { 1 \leq j \leq n _ { i } } \sqrt { A _ { i , j } } , \qquad D _ { i } = n _ { i } , } } \end{array}\tag{17}
$$

where $X _ { i , c }$ is the cth radiometrically corrected native spectral band before model normalization and padding, $\langle \cdot \rangle _ { p }$ denotes its spatial mean, and $A _ { i , j }$ is the annotated oriented-box area. Thus, $L _ { i }$ averages the spatial means of all eight bands, $S _ { i }$ is the perimage median ground-truth object scale, and $D _ { i }$ is the instance count. Every image in the 4,885-image MODA test set contains at least one valid annotated instance, so $S _ { i }$ is defined for every image. We rank all 4,885 images independently by $L _ { i } , S _ { i }$ , and $D _ { i }$ with filename tie-breaking. The lowest 20%, lowest 20%, and highest 20% form the low-illumination, small-object, and high-density subsets, respectively (977 images each). Across these subsets, SGLDet outperforms OSSDet, with its largest gains under low illumination (3.7% $\mathrm { \ m A P _ { 5 0 } }$ and 2.5% mAP).

TABLE II  
CHALLENGING-SUBSET COMPARISON ON MODA.
<table><tr><td rowspan="2">Subset</td><td colspan="3">OSSDet</td><td colspan="3">SGLDet (Ours)</td></tr><tr><td> $\mathbf { \hat { m } A P _ { 5 0 } }$  1</td><td> $\mathbf { \hat { m A P } 7 5 }$ </td><td>↑mAP↑</td><td> $\mathbf { \hat { m } A P _ { 5 0 } }$  ←</td><td> $\mathbf { \dot { m } A P } _ { 7 5 }$ </td><td>↑mAP↑</td></tr><tr><td>Full test</td><td>69.0</td><td>45.9</td><td>42.7</td><td>71.8</td><td>47.0</td><td>44.3</td></tr><tr><td>Low illum.</td><td>61.4</td><td>38.2</td><td>35.5</td><td>65.1</td><td>40.8</td><td>38.0</td></tr><tr><td>Small objects</td><td>56.2</td><td>32.5</td><td>30.8</td><td>59.0</td><td>34.8</td><td>33.0</td></tr><tr><td>High density</td><td>63.8</td><td>41.0</td><td>38.5</td><td>67.0</td><td>43.5</td><td>40.8</td></tr></table>

![](images/0df2a7ffe252c3edf48a2f562962e752ecfc9c2a1d9ce570341e596e4fb49702.jpg)  
Ground Truth  
OSSDet Feat  
SGLDet Feat (Ours)  
Ori. RepPoints Result

SGLDet Result (Ours)  
Fig. 5. Visualization comparison of detection results and feature maps obtained by different methods on MODA.  
![](images/b20215ff6aa511fc6504fc18f88fcd5c12631304a96d228ee1f96874b5966215.jpg)

![](images/01fa832a6aa74b48de63506e233498698951a49eded7499a670e0926e26cb775.jpg)  
Fig. 6. Two-dimensional t-SNE visualization of feature distributions learned by OSSDet and SGLDet on MODA.

![](images/26da6bd347399accee2d83a3eaa7d41609f189398bea11132b2c993de9d342cd.jpg)  
Fig. 7. Effectiveness visualization of key components.

Efficiency Analysis. As Table I shows, two-stage methods offer high accuracy but complex architectures, while conventional one-stage detectors are efficient yet less accurate. The MSIoriented S2ADet adopts a two-stream design to decouple spatial and spectral cues, which incurs substantial preprocessing and computational cost. In contrast, SGLDet achieves the highest $\mathrm { \ m A P _ { 5 0 } }$ with competitive computational cost, striking a favorable accuracy-efficiency balance.

Qualitative Results. As depicted in Figure 5, our method enhances the network’s focus on target regions while suppressing irrelevant background noise, even when objects are small or captured under low-visibility, low-illumination, and cluttered conditions. Fewer false positives and missed detections are observed. These improvements are consistent with the integration of spectrally guided global retrieval and input-adaptive local modulation, which jointly enhance the classification and localization of small objects with similar spectral signatures in complex backgrounds.

Visualization of Feature Distribution. T-SNE [35] visualization results are shown in Figure 6. Compared with OSSDet, the features extracted by SGLDet form more compact clusters within the same category, indicating enhanced intraclass consistency. Meanwhile, SGLDet shows wider inter-class separation for easily confused categories (e.g., car and van), small objects (e.g., pedestrian and bike), and few-sample categories (e.g., tricycle). This visualization supports the role of spectral-driven global dynamic modeling in making spectrally similar objects more semantically coherent while improving inter-class discriminability through adaptive global– local modulation.

Effect of Components. Figure 7 visualizes features involved in key components of SGLDet. (a) SGDM reinforces semantic coherence among spectrally similar objects through global retrieval of the most relevant spectral responses, thereby enhancing the responses of such objects across the scene. (b) MGCI integrates multi-scale features and establishes holistic scene perception through global contextual reasoning, strengthening the contrast between foreground objects and background regions. (c) SICA performs scene-guided local expert aggregation to adaptively refine local responses, enabling more discriminative local aggregation for fine-grained detection.

## B. Ablation Studies and Analysis

To evaluate the practical efficiency, key components, and compatibility of SGLDet with existing detectors, we conduct a series of analysis and ablation experiments on the MODA dataset. Unless otherwise specified, all studies follow the same MODA setting used in the main comparison.

Measured Inference Efficiency. Table III reports detection accuracy, FLOPs, inference latency, and FPS. All methods are benchmarked on a single idle NVIDIA RTX 3090 GPU with batch size 1, FP32 inference, and no test-time augmentation. Each detector uses the test-time post-processing configuration adopted for its MODA accuracy evaluation, including the score threshold and NMS settings where applicable. Native 1200 × 900 inputs are normalized and padded to $1 2 1 6 \times 9 2 8$ . Latency includes network forward propagation, prediction decoding, and NMS where applicable, while excluding I/O and memory copy operations. After 50 warm-up iterations, all detectors are timed on the same fixed subset of 500 MODA test images in each of three independent runs. CUDA is synchronized at both timing boundaries; latency is reported as the mean ± standard deviation of the three run means, and FPS is the reciprocal of the overall mean latency. SGLDet runs at 54 ms per image (18.5 FPS). It is 2 ms faster than Ori. RepPoints (56 ms, 17.9 FPS) while improving $\mathrm { \ m A P _ { 5 0 } }$ by 5.3 percentage points. Compared with OSSDet, SGLDet incurs a marginal 1- ms latency increase (1.9%) while improving m $\mathrm { { A P } _ { 5 0 } }$ and mAP by 2.8 and 1.6 percentage points, respectively. SGDM requires Top-K neighbor selection and feature gathering, whose runtime costs are not directly reflected in FLOPs; fewer FLOPs need not imply lower latency [36]. This controlled comparison shows that SGLDet achieves a favorable overall balance between detection accuracy and measured inference efficiency.

TABLE III  
MEASURED INFERENCE EFFICIENCY AND FLOATING-POINT OPERATIONCOUNT ON MODA.
<table><tr><td>Method</td><td> $\bf { m A P } _ { 5 0 }$ </td><td></td><td></td><td>↑ mAP↑ FLOPs↓ ms/img↓ FPS↑</td><td></td></tr><tr><td>O-RCNN [21]</td><td>65.5</td><td>40.9</td><td>244.7G</td><td> $5 1 \pm 0 . 7$ </td><td>19.6</td></tr><tr><td>Rot. Retina [23]</td><td>53.4</td><td>34.1</td><td>233.5G</td><td> $4 2 \pm 0 . 6$ </td><td>23.8</td></tr><tr><td>Ori. RepPoints [11]</td><td>66.5</td><td>40.9</td><td>213.6G</td><td> $5 6 { \pm } 0 . 8 $ </td><td>17.9</td></tr><tr><td>Ori-DETR [32]</td><td>67.8</td><td>41.9</td><td>492.0G</td><td> $7 6 { \pm } 1 . 5$ </td><td>13.2</td></tr><tr><td>S2ADet [1]</td><td>63.5</td><td>38.9</td><td>406.0G</td><td> $1 1 2 { \pm } 1 . 8 $ </td><td>8.9</td></tr><tr><td>OSSDet [2]</td><td>69.0</td><td>42.7</td><td>263.1G</td><td> $5 3 { \pm } 0 . 8 $ </td><td>18.9</td></tr><tr><td>SGLDet (Ours)</td><td>71.8</td><td>44.3</td><td>241.6G</td><td> $5 4 { \pm } 0 . 8 $ </td><td>18.5</td></tr></table>

Component Ablation Study. As shown in Table IV, SGDM and MGCI individually improve $\mathrm { \ m A P _ { 5 0 } }$ from 66.5% to 68.4% and 68.0%, respectively, while their combination reaches 70.2%. SGDM strengthens within-scale spectral consistency, whereas MGCI organizes the refined responses into cross-scale scene context. Adding SICA further raises $\mathrm { \ m A P _ { 5 0 } }$ to 71.8%, yielding a 5.3% improvement over the baseline and demonstrating the synergy among spectrally guided global retrieval, scene-level contextual reasoning, and input-adaptive local aggregation.

TABLE IV  
ABLATION STUDIES ON THE KEY COMPONENTS OF SGLDET.
<table><tr><td>SGDM</td><td>MGCI</td><td>SICA</td><td> $\mathrm { \ m A P _ { 5 0 } }$  ↑</td><td> $\mathrm { \ m A P _ { 7 5 } }$  ↑</td><td>mAP↑</td></tr><tr><td>x</td><td>x</td><td>x</td><td>66.5</td><td>44.1</td><td>40.9</td></tr><tr><td>√</td><td>x</td><td>x</td><td>68.4</td><td>44.5</td><td>41.4</td></tr><tr><td>X</td><td>√</td><td>x</td><td>68.0</td><td>44.6</td><td>41.5</td></tr><tr><td>x</td><td>x</td><td>√</td><td>68.3</td><td>44.8</td><td>41.7</td></tr><tr><td>√</td><td>√</td><td>x</td><td>70.2</td><td>45.8</td><td>42.6</td></tr><tr><td>√</td><td>J</td><td>√</td><td>71.8</td><td>47.0</td><td>44.3</td></tr></table>

TABLE V  
ABLATION STUDY OF THE TOP-K RATIO IN SGDM.
<table><tr><td>r 一</td><td> $\mathrm { \ m A P _ { 5 0 } }$  ←</td><td> $\mathrm { m A P _ { 7 5 } ~ } \uparrow$ </td><td>mAP↑</td></tr><tr><td>0.2</td><td>70.6</td><td>45.7</td><td>42.5</td></tr><tr><td>0.4</td><td>71.4</td><td>46.4</td><td>43.2</td></tr><tr><td>0.6</td><td>71.8</td><td>47.0</td><td>44.3</td></tr><tr><td>0.8</td><td>71.0</td><td>46.1</td><td>42.9</td></tr></table>

Ablation Study on SGDM. A key design in SGDM is the spectral consensus mechanism, which aligns each query with its spectrally nearest neighborhood before contextual aggregation. As shown in Table VI, removing this consensus step and directly aggregating retrieved values leads to a 0.6% drop in $\mathrm { \ m A P _ { 5 0 } }$ and a 1.0% drop in mAP, confirming that explicit query-side consensus formation strengthens spectral consistency and is beneficial for subsequent content retrieval. We further evaluate the Top-K selection ratio r that controls the sparsity of spectral retrieval. As shown in Table V, a small r restricts global interaction and may fail to capture sufficient spectrally consistent cues, while a large r introduces redundant or irrelevant features that dilute the spectral specificity of target tokens. SGLDet achieves the best results when $r = 0 . 6 $ , striking a balance between retrieval coverage and noise suppression.

We compare the negative squared Euclidean score with cosine similarity $\Psi _ { i j } ^ { \mathrm { c o s } } = { Q _ { i } ^ { \top } K _ { j } } \mathrm { \bar { / } ( | | } Q _ { i } | | _ { 2 } | | K _ { j } | | _ { 2 } )$ and negative SAM $\Psi _ { i j } ^ { \mathrm { S A M } } = - \operatorname { a r c c o s } ( \mathrm { c l i p } ( \Psi _ { i j } ^ { \mathrm { c o s } } , - 1 + \epsilon , 1 - \epsilon ) )$ , where ϵ is a small numerical constant and clip(·) bounds its input to the displayed interval for numerical stability. All other settings are fixed, and larger scores indicate closer neighbors in every variant. As shown in Table VII, the negative squared Euclidean score achieves the highest mAP of 44.3%, outperforming cosine similarity and negative SAM by 0.8% and 1.2%, respectively. These results support its use as SGDM’s affinity function.

Ablation Study on MGCI. MGCI uses a slot-writing stage to consolidate cross-scale tokens and a context-readout stage to retrieve position-adaptive scene context. Table VIII first studies the number of scene slots. Increasing M from 8 to 16 improves all three metrics, whereas using 32 slots causes a slight decline. This suggests that too few slots limit contextual capacity, while too many may introduce redundant scene representations. We therefore use $M = 1 6$ to balance context capacity and memory compactness. Table IX then compares the complete slot write– read path with a direct-MHCA variant that removes slot writing and lets the P scene-content queries attend directly to all N scene tokens. Compared with direct MHCA, the complete design improves $\mathrm { m A P _ { 5 0 } , \ m A P _ { 7 5 } }$ , and mAP by 0.6%, 0.6%, and 1.2%, respectively, supporting the use of compact scene memory for cross-scale context aggregation.

TABLE VI  
ABLATION STUDY ON SPECTRAL CONSENSUS IN SGDM.
<table><tr><td>Variant</td><td> $\mathrm { \ m A P { 5 0 } }$  ↑</td><td> $\mathrm { \ m A P _ { 7 5 } }$  ↑</td><td> $\mathrm { m A P \uparrow }$ </td></tr><tr><td>w/o spectral consensus</td><td>71.2</td><td>46.5</td><td>43.3</td></tr><tr><td>w/ spectral consensus</td><td>71.8</td><td>47.0</td><td>44.3</td></tr></table>

TABLE VIII  
ABLATION STUDY OF THE SCENE-MEMORY CARDINALITY IN MGCI.
<table><tr><td>M</td><td> $\mathrm { \ m A P _ { 5 0 } }$  ↑</td><td> $\mathrm { \ m A P _ { 7 5 } }$  ←</td><td> $\mathrm { m A P \uparrow }$ </td></tr><tr><td>8</td><td>71.3</td><td>46.2</td><td>43.1</td></tr><tr><td>16</td><td>71.8</td><td>47.0</td><td>44.3</td></tr><tr><td>32</td><td>71.5</td><td>46.6</td><td>43.4</td></tr></table>

TABLE X  
ABLATION STUDY ON THE KEY DESIGN CHOICES OF SICA.
<table><tr><td>Variant</td><td> $\mathrm { \ m A P _ { 5 0 } }$  ←</td><td> $\mathrm { \ m A P _ { 7 5 } }$  ←</td><td> $\mathrm { m A P \uparrow }$ </td></tr><tr><td>Single Expert  $( M { = } 1 )$ </td><td>70.4</td><td>45.8</td><td>42.7</td></tr><tr><td>w/o Channel Calibration</td><td>71.3</td><td>46.5</td><td>43.4</td></tr><tr><td>w/o Spatial Calibration</td><td>71.1</td><td>46.3</td><td>43.2</td></tr><tr><td>Full SICA</td><td>71.8</td><td>47.0</td><td>44.3</td></tr></table>

Ablation Study on SICA. SICA performs scene-guided local expert aggregation through a learnable expert bank together with lightweight channel-spatial calibration. We conduct two groups of experiments to validate these design choices. First, Table X examines the contribution of expert diversity and sceneguided calibration. Replacing the expert bank with a single expert (M=1) reduces $\mathrm { \ m A P _ { 5 0 } }$ by 1.4% and mAP by 1.6%, indicating that a shared local filter is insufficient to capture the heterogeneous local patterns of aerial objects. Removing chan nel calibration or spatial calibration also degrades performance, confirming that global scene cues provide complementary guidance for adaptive local refinement beyond the routed expert response alone. We further evaluate the number of experts in Table XI. Using too few experts limits the diversity of local aggregation patterns, whereas an excessively large expert bank introduces redundancy and slightly weakens generalization. The best trade-off is obtained with M=8, which is therefore used in all experiments.

Compatibility Analysis. To assess compatibility with existing detectors, we integrate SGDM into the last three stages of the R50 backbone in five representative models; the baseline rows are taken from Table I. As shown in Table XII, SGDM improves every baseline, with m $\mathsf { A P } _ { 5 0 }$ gains ranging from 1.1% to 1.6%. The gains across different detector families support its use as a broadly compatible backbone enhancement rather than a head-specific modification.

## C. Results on the HOD3K Dataset

Data and Setups. The HOD3K dataset [1] is a multi-band spectral object detection benchmark collected from urban roads, campuses, and residential areas. It contains 3,242 images with

TABLE VII  
ABLATION STUDY OF THE AFFINITY FUNCTION IN SGDM.
<table><tr><td>Affinity function</td><td> $\mathrm { \ m A P _ { 5 0 } }$  ↑</td><td> $\mathrm { \ m A P _ { 7 5 } }$ </td><td>↑ mAP↑</td></tr><tr><td>Cosine similarity</td><td>71.1</td><td>46.3</td><td>43.5</td></tr><tr><td>Spectral Angle Mapper</td><td>70.7</td><td>46.0</td><td>43.1</td></tr><tr><td>Euclidean affinity</td><td>71.8</td><td>47.0</td><td>44.3</td></tr></table>

TABLE IX

ABLATION STUDY ON THE KEY DESIGN CHOICES OF MGCI.
<table><tr><td>Compact Memory Scene Readout</td><td></td><td> $\mathrm { \ m A P { 5 0 } }$  ←</td><td> $\mathrm { \ m A P _ { 7 5 } }$ </td><td>↑mAP↑</td></tr><tr><td>X</td><td>x</td><td>70.3</td><td>45.7</td><td>42.6</td></tr><tr><td>√</td><td>x</td><td>71.0</td><td>46.2</td><td>43.0</td></tr><tr><td>x</td><td>V</td><td>71.2</td><td>46.4</td><td>43.1</td></tr><tr><td>V</td><td>√</td><td>71.8</td><td>47.0</td><td>44.3</td></tr></table>

TABLE XI

ABLATION STUDY OF THE NUMBER OF EXPERTS IN SICA.
<table><tr><td>M</td><td> $\mathrm { \ m A P _ { 5 0 } }$  ←</td><td> $\mathrm { \ m A P _ { 7 5 } ~ } ^ { \cdot }$  个</td><td>mAP↑</td></tr><tr><td>4</td><td>70.3</td><td>45.8</td><td>42.7</td></tr><tr><td>8</td><td>71.8</td><td>47.0</td><td>44.3</td></tr><tr><td>12</td><td>71.4</td><td>46.7</td><td>43.5</td></tr><tr><td>16</td><td>71.2</td><td>46.5</td><td>43.3</td></tr></table>

TABLE XII  
COMPATIBILITY ANALYSIS ON REPRESENTATIVE R50-BASED DETECTORS ON MODA. BASELINE RESULTS ARE TAKEN FROM TABLE I.
<table><tr><td>Detector</td><td>Backbone</td><td> $\bf { m A P } _ { 5 0 }$  ↑</td><td> $\mathbf { m A P } _ { 7 5 }$  ↑</td><td>mAP↑</td></tr><tr><td rowspan="2">O-RCNN</td><td>R50</td><td>65.5</td><td>44.0</td><td>40.9</td></tr><tr><td> $\mathbf { R 5 0 _ { S G D M } }$  一</td><td> ${ \bf 6 7 . 1 } ( + 1 . 6 )$ </td><td></td><td>45.2(+1.2) 41.9(+1.0)</td></tr><tr><td rowspan="2">Rot. FCOS</td><td>R50</td><td>63.5</td><td>41.8</td><td>39.1</td></tr><tr><td>R50SGDM</td><td>64.9(+1.4) 42.9(+1.1) 40.0(+0.9)</td><td></td><td></td></tr><tr><td rowspan="2">CFA</td><td>R50</td><td>66.2</td><td>43.2</td><td>40.6</td></tr><tr><td>R50SGDM</td><td>67.7(+1.5) 44.3(+1.1) 41.6(+1.0)</td><td></td><td></td></tr><tr><td rowspan="2">Ori. RepPoints</td><td>R50</td><td>66.5</td><td>44.1</td><td>40.9</td></tr><tr><td> $\mathbf { R 5 0 _ { S G D M } }$  一</td><td> ${ \bf 6 7 . 8 ( + 1 . 3 ) }$ </td><td></td><td>45.1(+1.0) 41.8(+0.9)</td></tr><tr><td rowspan="2">Ori-DETR</td><td>R50</td><td>67.8</td><td>45.1</td><td></td></tr><tr><td> $\mathbf { R 5 0 _ { S G D M } }$ </td><td>68.9(+1.1) 45.9(+0.8) 42.7(+0.8)</td><td></td><td>41.9</td></tr></table>

![](images/8ddad8060ce09ac48112f41800d5c332ab289d3376d90f58994e82c65baf32be.jpg)  
Fig. 8. Detection results on HOD3K. Red boxes indicate the ground truth, while green boxes denote the predictions.

TABLE XIII  
COMPARISON WITH STATE-OF-THE-ART DETECTORS ON HOD3K.
<table><tr><td>Method</td><td> $\mathbf { \left| m A P _ { 5 0 } \right| }$  ←  $\mathbf { m A P } _ { 7 5 }$ </td><td></td><td></td><td>↑ mAP↑|FLOPs↓ Params↓</td></tr><tr><td>D-DETR [7]</td><td>85.0</td><td>52.0</td><td>48.8</td><td>109.6G 40.1M</td></tr><tr><td>CO-DETR [37]</td><td>77.9</td><td>57.3</td><td>51.7 200.7G</td><td>64.5M</td></tr><tr><td>RetinaNet [23]</td><td>91.2</td><td>59.4</td><td>53.3 205.7G</td><td>36.2M</td></tr><tr><td>YOLOv5 [38]</td><td>88.1</td><td>59.8</td><td>54.4 48.3G</td><td>20.9M</td></tr><tr><td>DINO [39]</td><td>89.1</td><td>61.6</td><td>56.3 114.7G</td><td>47.6M</td></tr><tr><td>Fovea [40]</td><td>92.2</td><td>62.2</td><td>56.8 123.9G</td><td>38.0M</td></tr><tr><td>Faster RCNN [41]</td><td>89.4</td><td>62.7</td><td>56.9 206.7G</td><td>41.1M</td></tr><tr><td>ATSS [42]</td><td>91.2</td><td>63.0</td><td>57.4 110.6G</td><td>32.2M</td></tr><tr><td>TOOD [43]</td><td>90.3</td><td>65.6</td><td>57.8</td><td>108.5G 32.1M</td></tr><tr><td>Meth. Mapper [44]</td><td>88.7</td><td>63.8</td><td>57.9</td><td>267.6G 80.3M</td></tr><tr><td>RepPoints [25]</td><td>92.0</td><td>65.5</td><td>58.4</td><td>104.5G 36.9M</td></tr><tr><td>PAA [45]</td><td>91.2</td><td>65.0</td><td>58.4 110.6G</td><td>32.2M</td></tr><tr><td>VFNet [46]</td><td>92.3</td><td>66.5</td><td>59.0</td><td>104.5G 32.8M</td></tr><tr><td>C-RCNN [6]</td><td>90.9</td><td>66.9</td><td>59.6</td><td>149.5G 69.2M</td></tr><tr><td>S2ADet [1]</td><td>93.4</td><td></td><td>59.8</td><td>169.2G 48.6M</td></tr><tr><td>OSSDet [2]</td><td>93.4</td><td>68.8</td><td>60.9</td><td>136.2G 37.7M</td></tr><tr><td>SGLDet (Ours)</td><td>94.0</td><td>69.4</td><td>61.3</td><td>127.8G 37.2M</td></tr></table>

![](images/798f82adc1f6b90594ec451fa8a61a4bb918342dbfdb835dd661ba0d77848f78.jpg)  
Fig. 9. Detection results on DroneVehicle. Red boxes indicate the ground truth, while green boxes denote the predictions.

16 spectral bands from 470 to 620 nm and a spatial resolution of $5 1 2 \times 2 5 6$ . The dataset provides 15,149 annotated instances from three categories (people, cars, and bikes) and is split into 2,308 training, 219 validation, and 715 testing images. All 16 spectral bands were used as input, and training was conducted for 12 epochs with a total batch size of 8 at an input resolution of $1 0 2 4 \times 5 1 2$ . Other optimization settings followed those used on MODA, while axis-aligned IoU was adopted for evaluation. Quantitative Results. As shown in Table XIII, SGLDet achieves state-of-the-art performance across all metrics, outperforming the second-best method by 0.6%, 0.6%, and 0.4% in $\mathrm { m A P _ { 5 0 } , \ m A P _ { 7 5 } }$ , and mAP, respectively. S2ADet obtains competitive accuracy via a dual-stream spectral–spatial decoupling design, but incurs additional preprocessing and computational overhead. OSSDet directly processes MSI data and leverages object-aware guidance to learn target-relevant spectral–spatial features, improving efficiency while maintaining strong performance. In contrast, SGLDet conducts unified global and local dynamic modeling under joint spectral–spatial guidance. With cost comparable to OSSDet, it achieves higher accuracy, demonstrating more effective spectral–spatial utilization and a better efficiency–performance trade-off. Small, scalevarying aerial targets [47], [48] and low-light conditions [49] complicate detection. On HOD3K, with larger relative targets and more stable backgrounds [2], SGLDet’s improvement is less pronounced than on MODA.

Qualitative Results. As shown in Figure 8, existing methods often fail to reliably detect small objects embedded in visually similar backgrounds. This limitation stems from insufficient exploitation of spectral–spatial cues under challenging conditions. In contrast, SGLDet enhances target discrimination through unified global–local dynamic modeling guided by spectral–spatial cues, enabling more effective separation of foreground objects from background clutter and producing results that align closely with ground truth.

## D. Results on the DroneVehicle Dataset

Data and Setups. The DroneVehicle dataset [4] is a widely used RGB-infrared aerial benchmark for drone-based vehicle detection. It contains 28,439 paired visible–infrared images and 953,087 oriented annotations over 5 vehicle categories, with 17,990 training pairs, 1,469 validation pairs, and 8,980 testing pairs under the official split. Instead of using two-stream fusion, the original SGLDet concatenates each RGB-infrared pair along the channel dimension as a unified four-channel spectral input within the same single-stream framework. This design exploits complementary cues from both modalities through a shared encoder without introducing a separate infrared branch. The input image size was set to $6 4 0 \times 5 1 2 .$ , and the model was trained for 12 epochs with a total batch size of 16. Other optimization settings followed those used on MODA.

Quantitative Results. Table XIV shows that the original four-channel model achieves 81.5% $\mathrm { \ m A P _ { 5 0 } }$ and 61.8% mAP, exceeding MS2Fusion by 1.3% and 1.4%, respectively. The dual stem applies separate $7 \times 7$ convolutions to RGB and infrared, concatenates their outputs, and projects them to 64 channels with a $1 \times 1$ convolution before the shared backbone. The original model leads on bus; the dual stem leads on truck and freight car, while IV-YOLO and DMM lead on car and van. Early fusion improves all five class scores over both singlemodal settings. Against RGB-only, it gains 2.1% in $\mathrm { \ m A P _ { 5 0 } }$ and 3.0% in mAP. Against IR-only, it gains 1.1% in mAP<sub>50</sub> and 1.5% in mAP. The dual stem reaches 81.7% $\mathrm { \ m A P _ { 5 0 } }$ and 61.9% mAP, adding 0.2% and 0.1%, respectively. Prior RGB– infrared detection comparisons also found that fusing modality features after extraction performed slightly better than input fusion [50], [51].

Qualitative Results. As illustrated in Figure 9, representative RGB-infrared detectors still exhibit missed detections and false responses in densely distributed traffic scenes, especially when vehicles are small, heavily overlapped, or affected by illumination variation across modalities. In contrast, SGLDet produces more complete and accurate predictions under these challenging conditions. This qualitative advantage is consistent with the proposed global-local dynamic modeling strategy, which couples spectrally guided global reasoning with sceneaware local refinement to better preserve target completeness while suppressing cross-modal background interference.

TABLE XIV PERFORMANCE COMPARISONS ON DRONEVEHICLE.
<table><tr><td>Method</td><td>[Car Bus Truck Fre. Van|</td><td></td><td>mAP50 ↑ mAP↑</td><td></td></tr><tr><td>C²Former [52] YOLOv81-RGB [53] YOLOv81-IR [53] SLBAF-Net [54]</td><td>90.2 89.8 68.3 92.5 91.7 68.8 93.4 91.9 69.3</td><td>64.4 58.5 47.8 50.2 53.7 51.1</td><td>74.2 70.2 71.9</td><td>47.3 48.6 49.1</td></tr><tr><td>CSOM-ODAF [55] Multi. DINO [56]</td><td>90.2 89.9 90.1 89.8 89.5 88.8</td><td>76.0 68.2 59.9 75.6 68.2 61.8 75.4 54.3 54.3</td><td>76.8 77.1 72.5</td><td>49.5 50.1 50.3</td></tr><tr><td>CMA [57]</td><td>95.8 93.1 75.9</td><td>59.8 59.4</td><td>76.8</td><td>50.4</td></tr><tr><td>CRSIOD [58]</td><td>95.6 92.2 71.7</td><td>50.5 55.8</td><td>73.2</td><td>50.8</td></tr><tr><td></td><td>90.3 88.0 72.7</td><td>53.6 52.6</td><td></td><td></td></tr><tr><td>GLFNet [59]</td><td></td><td></td><td>71.4</td><td>54.8</td></tr><tr><td>Dual-YOLO [60]</td><td>95.9 91.6 69.7</td><td>55.9 46.6</td><td>71.9</td><td>55.2</td></tr><tr><td>DMM [61]</td><td>90.4 88.7 77.8</td><td>63.0 66.0</td><td>77.2</td><td>55.8</td></tr><tr><td>IV-YOLO [62]</td><td>97.2 94.3 65.4</td><td>63.1 53.0</td><td>74.6</td><td>56.8</td></tr><tr><td>WaveMamba [63]</td><td>95.0 90.6 80.4</td><td>68.5 64.5</td><td>79.8</td><td>60.5</td></tr><tr><td>O-RCNN [21]</td><td>90.8 90.0 66.8</td><td>60.2 50.2</td><td>71.6</td><td>44.3</td></tr><tr><td>S2ANet [10]</td><td>90.6 89.5 66.2 78.0</td><td>61.8 52.9</td><td>72.2</td><td>43.8</td></tr><tr><td>OSSDet [2]</td><td>94.0 89.5</td><td>67.0 63.0</td><td>78.3</td><td>58.5</td></tr><tr><td>ICAFusion [64]</td><td>95.6 92.3 77.8</td><td>64.5 60.3</td><td>78.1</td><td>58.6</td></tr><tr><td>SFFR [65]</td><td>95.8 93.1 79.2 80.5</td><td>66.1 62.3</td><td>79.3</td><td>59.4</td></tr><tr><td>MS2Fusion [66]</td><td>96.1 93.7 95.7 92.2</td><td>67.8 62.9</td><td>80.2</td><td>60.4</td></tr><tr><td>IRDFusion [67]</td><td>79.4</td><td>66.3 64.1</td><td>79.5</td><td>59.7</td></tr><tr><td></td><td>|95.8 93.9 78.9</td><td>65.5 62.9|</td><td></td><td></td></tr><tr><td>SGLDet-RGB</td><td>95.6 93.8 80.3</td><td>67.9 64.4</td><td>79.4</td><td>58.8</td></tr><tr><td>SGLDet-IR</td><td></td><td></td><td>80.4</td><td>60.3</td></tr><tr><td>SGLDet</td><td>96.5 94.8 81.2</td><td>69.7 65.5</td><td>81.5</td><td>61.8</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>96.6 94.7</td><td>69.9 65.8</td><td>81.7</td><td></td></tr><tr><td>SGLDet-DualStem</td><td>81.5</td><td></td><td></td><td>61.9</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

## V. CONCLUSION

We propose SGLDet, a spectral-aware global-to-local dynamic modeling framework for multispectral aerial object detection. To address unreliable recognition caused by severe foreground-background imbalance and inaccurate localization caused by substantial variations in target scale, orientation, and structure, SGLDet integrates three complementary modules. SGDM captures long-range spectral dependencies through similarity-guided sparse retrieval to improve semantic consistency among spectrally similar objects. MGCI consolidates multi-scale evidence into a compact scene memory for holistic scene understanding. SICA further performs dynamic feature modulation via input-adaptive expert aggregation and spectral spatial calibration to handle pronounced local variations. Extensive experiments on MODA, HOD3K, and DroneVehicle demonstrate the effectiveness and generalization ability of SGLDet across diverse multispectral detection scenarios. Future work will explore extending SGLDet’s global-to-local spectral spatial modeling to other multispectral aerial perception tasks.

## VI. ACKNOWLEDGMENTS

This work was financially supported by the National Natural Science Foundation of China (No. 62101032), the Chongqing Excellent Young Scientists Fund (No. CSTB2025NSCQ-JQX0017), and the High-Quality Development Special Project of the Ministry of Industry and Information Technology (TC240HAJ9-35).

## REFERENCES

[1] X. He, C. Tang, X. Liu, W. Zhang, K. Sun, and J. Xu, “Object detection in spectral image via unified spectral-spatial feature aggregation,” IEEE Transactions on Geoscience and Remote Sensing, 2023.

[2] S. Han, T. Xu, P. Liu, and J. Li, “Moda: The first challenging benchmark for multispectral object detection in aerial images,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 40, no. 6, 2026, pp. 4574–4582.

[3] M. Xu, H. Zhang, W. He, S. Zhang, and S. Jia, “Dahnet: A decomposition and aggregation hybrid network for spectral object detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 64, p. 5506217, 2026.

[4] Y. Sun, B. Cao, P. Zhu, and Q. Hu, “Drone-based rgb-infrared crossmodality vehicle detection via uncertainty-aware learning,” arXiv preprint arXiv:2003.02437, 2020.

[5] Z. Tian, C. Shen, H. Chen, and T. He, “Fcos: Fully convolutional onestage object detection,” in ICCV. IEEE Computer Society, 2019, pp. 9626–9635.

[6] Z. Cai and N. Vasconcelos, “Cascade r-cnn: High quality object detection and instance segmentation,” IEEE Transactions on Pattern Analysis and Machine Intelligence, p. 1–1, 2019.

[7] X. Zhu, W. Su, L. Lu, B. Li, X. Wang, and J. Dai, “Deformable detr: Deformable transformers for end-to-end object detection,” in ICLR, 2021.

[8] J. Ding, N. Xue, Y. Long, G.-S. Xia, and Q. Lu, “Learning roi transformer for oriented object detection in aerial images,” in CVPR, 2019, pp. 2849– 2858.

[9] X. Yang, J. Yan, Z. Feng, and T. He, “R3det: Refined single-stage detector with feature refinement for rotating object,” in AAAI, vol. 35, no. 4, 2021, pp. 3163–3171.

[10] J. Han, J. Ding, J. Li, and G.-S. Xia, “Align deep features for oriented object detection,” IEEE transactions on geoscience and remote sensing, vol. 60, pp. 1–11, 2021.

[11] W. Li, Y. Chen, K. Hu, and J. Zhu, “Oriented reppoints for aerial object detection,” in CVPR, 2022, pp. 1829–1838.

[12] Z. Guo, C. Liu, X. Zhang, J. Jiao, X. Ji, and Q. Ye, “Beyond boundingbox: Convex-hull feature adaptation for oriented and densely packed object detection,” in CVPR, 2021, pp. 8792–8801.

[13] Y. Li, Q. Hou, Z. Zheng, M. Cheng, J. Yang, and X. Li, “Lsknet: A foundation lightweight backbone for remote sensing,” International Journal of Computer Vision, vol. 132, pp. 3736–3752, 2024.

[14] X. Yuan, Z. Zheng, Y. Li, X. Liu, L. Liu, X. Li, Q. Hou, and M.-M. Cheng, “Strip r-cnn: Large strip convolution for remote sensing object detection,” arXiv preprint arXiv:2501.03775, 2025.

[15] X. Wu, X. Wang, X. Zhu, L. Yang, J. Liu, and X. Hu, “Measuring the impact of rotation equivariance on aerial object detection,” in ICCV, 2025, accepted by ICCV 2025.

[16] D. Hong, Z. Han, J. Yao, L. Gao, B. Zhang, A. Plaza, and J. Chanussot, “Spectralformer: Rethinking spectral image classification with transformers,” IEEE Transactions on Geoscience and Remote Sensing, vol. 60, pp. 1–15, 2021.

[17] L. Yan, M. Zhao, X. Wang, Y. Zhang, and J. Chen, “Object detection in spectral images,” IEEE Signal Processing Letters, vol. 28, pp. 508–512, 2021.

[18] F. A. Kruse, A. B. Lefkoff, J. W. Boardman, K. B. Heidebrecht, A. T. Shapiro, P. J. Barloon, and A. F. H. Goetz, “The spectral image processing system (SIPS)—interactive visualization and analysis of imaging spectrometer data,” Remote Sensing of Environment, vol. 44, no. 2–3, pp. 145–163, 1993.

[19] F. D. van der Meer, “The effectiveness of spectral similarity measures for the analysis of hyperspectral imagery,” International Journal of Applied Earth Observation and Geoinformation, vol. 8, no. 1, pp. 3–17, 2006.

[20] Y. Zhang, E. Yang, H. H. Yoon, Q. Cheng, Z. Sun, T. Hasan, and W. Cai, “Reconstructive spectrometers: Hardware miniaturization and computational reconstruction,” eLight, vol. 5, 2025, art. no. 23.

[21] X. Xie, G. Cheng, J. Wang, X. Yao, and J. Han, “Oriented r-cnn for object detection,” in ICCV, 2021, pp. 3520–3529.

[22] J. Han, J. Ding, N. Xue, and G.-S. Xia, “ReDet: A rotation-equivariant detector for aerial object detection,” in CVPR, 2021, pp. 2786–2795.

[23] T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, “Focal loss for´ dense object detection,” in ICCV, 2017, pp. 2980–2988.

[24] X. Yang, J. Yan, Q. Ming, W. Wang, X. Zhang, and Q. Tian, “Rethinking rotated object detection with gaussian wasserstein distance loss,” in ICML. PMLR, 2021, pp. 11 830–11 841.

[25] Z. Yang, S. Liu, H. Hu, L. Wang, and S. Lin, “Reppoints: Point set representation for object detection,” in ICCV, 2019, pp. 9657–9666.

[26] X. Yang, X. Yang, J. Yang, Q. Ming, W. Wang, Q. Tian, and J. Yan, “Learning high-precision bounding box for rotated object detection via kullback-leibler divergence,” NeurIPS, vol. 34, pp. 18 381–18 394, 2021.

[27] C. Lyu, W. Zhang, H. Huang, Y. Zhou, Y. Wang, Y. Liu, S. Zhang, and K. Chen, “RTMDet: An empirical study of designing real-time object detectors,” arXiv preprint arXiv:2212.07784, 2022.

[28] C. Xu, J. Ding, J. Wang, W. Yang, H. Yu, L. Yu, and G.-S. Xia, “Dynamic coarse-to-fine learning for oriented tiny object detection,” in CVPR, June

2023, pp. 7318–7328.

[29] L. Dai, H. Liu, H. Tang, Z. Wu, and P. Song, “AO2-DETR: Arbitraryoriented object detection transformer,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 5, pp. 2342–2356, 2022.

[30] Y. Zeng, Y. Chen, X. Yang, Q. Li, and J. Yan, “ARS-DETR: Aspect ratio-sensitive detection transformer for aerial oriented object detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–15, 2024.

[31] H. Lee, M. Song, J. Koo, and J. Seo, “Hausdorff distance matching with adaptive query denoising for rotated detection transformer,” in WACV, February 2025, pp. 1872–1882.

[32] Z. Zhao, Q. Xue, Y. He, Y. Bai, X. Wei, and Y. Gong, “Projecting points to axes: Oriented object detection via point-axis representation,” in ECCV, 2024, pp. 161–179.

[33] L. Huang, R. Luo, X. Liu, and X. Hao, “Spectral imaging with deep learning,” Light: Science & Applications, vol. 11, 2022, art. no. 61.

[34] C. Williams, R. Cousins, C. J. Mellor, S. E. Bohndiek, and G. S. D. Gordon, “Hyperpixels: Pixel filter arrays of multivariate optical elements for optimized spectral imaging,” Light: Advanced Manufacturing, vol. 6, 2025, art. no. 67.

[35] L. Van der Maaten and G. Hinton, “Visualizing data using t-sne.” Journal of machine learning research, vol. 9, no. 11, 2008.

[36] Z. Pan, J. Cai, and B. Zhuang, “Fast vision transformers with HiLo attention,” in Advances in Neural Information Processing Systems, vol. 35, 2022.

[37] Z. Zong, G. Song, and Y. Liu, “DETRs with collaborative hybrid assignments training,” in ICCV, 2023, pp. 6748–6758.

[38] G. Jocher, A. Chaurasia, A. Stoken, J. Borovec, Y. Kwon, K. Michael, J. Fang, Z. Yifu, C. Wong, D. Montes et al., “YOLOv5,” GitHub repository, 2020, accessed: 2026-04-21. [Online]. Available: https://github.com/ultralytics/yolov5

[39] H. Zhang, F. Li, S. Liu, L. Zhang, H. Su, J. Zhu, L. M. Ni, and H.- Y. Shum, “DINO: DETR with improved denoising anchor boxes for end-to-end object detection,” in ICLR, 2023.

[40] T. Kong, F. Sun, H. Liu, Y. Jiang, L. Li, and J. Shi, “Foveabox: Beyond anchor-based object detection,” IEEE Transactions on Image Processing, vol. 29, pp. 7389–7398, 2020.

[41] S. Ren, K. He, R. Girshick, and J. Sun, “Faster R-CNN: Towards realtime object detection with region proposal networks,” in Advances in Neural Information Processing Systems, 2015.

[42] S. Zhang, C. Chi, Y. Yao, Z. Lei, and S. Z. Li, “Bridging the gap between anchor-based and anchor-free detection via adaptive training sample selection,” in CVPR, 2020, pp. 9759–9768.

[43] C. Feng, Y. Zhong, Y. Gao, M. R. Scott, and W. Huang, “Tood: Taskaligned one-stage object detection,” in ICCV, 2021.

[44] S. Kumar, I. Arevalo, A. Iftekhar, and B. Manjunath, “MethaneMapper: Spectral absorption aware transformer for methane detection,” in CVPR, 2023, pp. 17 609–17 618.

[45] K. Kim and H. S. Lee, “Probabilistic anchor assignment with iou prediction for object detection,” in ECCV, 2020.

[46] H. Zhang, Y. Wang, F. Dayoub, and N. Sunderhauf, “Varifocalnet: An¨ iou-aware dense object detector,” in CVPR, 2021, pp. 8514–8523.

[47] P. Jia, D. Tian, Y. Wang, D. Yao, R. Xu, J. Sun, H. Sun, and B. Zhang, “Airborne optical imaging technology: A road map in CIOMP,” Light: Science & Applications, vol. 14, 2025, art. no. 119.

[48] J. Li, S. Jiang, L. Song, P. Peng, F. Mu, H. Li, P. Jiang, and T. Xu, “Automated optical inspection of FAST’s reflector surface using drones and computer vision,” Light: Advanced Manufacturing, vol. 4, no. 1, pp. 3–13, 2023.

[49] B. Jalali and C. MacPhee, “VEViD: Vision enhancement via virtual diffraction and coherent detection,” eLight, vol. 2, 2022, art. no. 24.

[50] S. Woo, K. Park, I. Shin, M. Kim, and I. S. Kweon, “MTMMC: A large-scale real-world multi-modal camera tracking benchmark,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 22 335–22 346.

[51] Z. Cao, H. Yang, J. Zhao, S. Guo, and L. Li, “Attention fusion for one-stage multispectral pedestrian detection,” Sensors, vol. 21, no. 12, p. 4184, 2021.

[52] M. Yuan and X. Wei, “C<sup>2</sup>Former: Calibrated and complementary transformer for RGB-infrared object detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–12, 2024.

[53] G. Jocher, A. Chaurasia, and J. Qiu, “YOLOv8,” Ultralytics, 2023, accessed: 2026-04-21. [Online]. Available: https://docs.ultralytics.com/ models/yolov8/

[54] X. Cheng, K. Geng, Z. Wang, J. Wang, Y. Sun, and P. Ding, “SLBAF-Net: Super-lightweight bimodal adaptive fusion network for UAV detection in low recognition environment,” Multimedia Tools and Applications, vol. 82, no. 30, pp. 47 773–47 792, 2023.

[55] C. Chen, J. Qi, X. Liu, K. Bin, R. Fu, X. Hu, and P. Zhong, “Weakly

misalignment-free adaptive feature alignment for UAVs-based multimodal object detection,” in CVPR, 2024, pp. 26 836–26 845.

[56] X. Sun, Y. Yu, and Q. Cheng, “Adaptive multimodal feature fusion with frequency domain gate for remote sensing object detection,” Remote Sensing Letters, vol. 15, no. 2, pp. 133–144, 2024.

[57] C. Jiang et al., “M2FNet: Multi-modal fusion network for object detection from visible and thermal infrared images,” International Journal of Applied Earth Observation and Geoinformation, vol. 130, p. 103918, 2024.

[58] H. Wang, C. Wang, Q. Fu, D. Zhang, R. Kou, Y. Yu, and J. Song, “Cross-modal oriented object detection of UAV aerial images based on image feature,” IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1–21, 2024.

[59] X. Kang, H. Yin, and P. Duan, “Global-local feature fusion network for visible-infrared vehicle detection,” IEEE Geoscience and Remote Sensing Letters, vol. 21, pp. 1–5, 2024.

[60] C. Bao, J. Cao, Q. Hao, Y. Cheng, Y. Ning, and T. Zhao, “Dual-YOLO architecture from infrared and visible images for object detection,” Sensors, vol. 23, no. 6, p. 2934, 2023.

[61] M. Zhou et al., “DMM: Disparity-guided multispectral mamba for oriented object detection in remote sensing,” 2024.

[62] D. Tian, X. Yan, D. Zhou, C. Wang, and W. Zhang, “IV-YOLO: A lightweight dual-branch object detection network,” Sensors, vol. 24, no. 19, p. 6181, 2024.

[63] H. Zhu et al., “WaveMamba: Wavelet-driven mamba fusion for RGBinfrared object detection,” in ICCV, 2025, pp. 11 219–11 229.

[64] J. Shen, Y. Chen, Y. Liu, X. Zuo, H. Fan, and W. Yang, “ICAFusion: Iterative cross-attention guided feature fusion for multispectral object detection,” Pattern Recognition, vol. 145, p. 109913, 2024.

[65] X. Zuo, C. Qu, H. Zhan, J. Shen, and W. Yang, “SFFR: Spatial–frequency feature reconstruction for multispectral aerial object detection,” IEEE Transactions on Geoscience and Remote Sensing, vol. 63, pp. 1–11, 2025.

[66] J. Shen, H. Zhan, S. Dong, X. Zuo, W. Yang, and H. Ling, “Multispectral state-space feature fusion: Bridging shared and cross-parametric interactions for object detection,” Information Fusion, vol. 127, p. 103895, 2026.

[67] J. Shen, H. Zhan, X. Zuo, H. Fan, X. Yuan, J. Li, and W. Yang, “IRDFusion: Iterative relation-map difference guided feature fusion for multispectral object detection,” Pattern Recognition, vol. 176, p. 113189, 2026.