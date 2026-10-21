# S<sup>3</sup>E-Det: A Strict Spatial–Spectral Equivariant Detector for Multispectral Object Detection

Peng Zhang, Tingfa Xu<sup>\*</sup>, Jinrui Sui, Yuyang Tian, and Jianan Li

Abstract—Multispectral object detection requires detectors to exploit material-dependent spectral responses while maintaining consistent localization under geometric variations. Existing MOD detectors learn discriminative spectral representations, yet lack a unified mechanism that preserves spectral order, adapts spectral– context fusion across pyramid levels, and regulates geometric responses under rotation. To address these limitations, we propose S<sup>3</sup>E-Det, a strict spatial–spectral equivariant detector that couples ordered spectral modeling with transformation-consistent prediction. Spectral Order-aware Sampling Calibration (SOSC) imposes local order constraints on input-derived spectral features through first-order variation, while Common-Differential Equivariant Fusion (CDEF) adapts spectral–context aggregation across pyramid levels. A compact Geometric Alignment Head (GAH) reconciles invariant and equivariant output properties within a unified prediction stage. Across MODA, HOD3K, and DroneVehicle, S<sup>3</sup>E-Det combines high detection accuracy with a compact parameter footprint and reduced rotation-induced variation. Code is available at https://github.com/Riiluo/S3E-Det.

Index Terms—remote sensing, multispectral object detection, rotation-equivariant neural networks, spectral order modeling, spatial-spectral feature learning

## I. INTRODUCTION

Multispectral images (MSIs) capture scene reflectance across multiple spectral bands, revealing material-dependent spectral signatures beyond RGB imagery [1]. Multispectral object detection (MOD) is a fundamental task for remote sensing scene understanding, with applications in traffic monitoring, urban management, and public safety [2], [3]. Existing MOD methods reduce spectral redundancy through band selection and principal-component analysis [7], or model cross-band dependencies through modulation, masking, and attention [3]. These operations improve feature discrimination but remain centered on channel selection and transformation, without explicitly modeling the local ordering of spectral bands. Their fusion designs likewise do not condition the balance between local differential spectral evidence and high-level object context on pyramid level. Thus, MOD requires spectral-order preservation and scale-adaptive fusion (Figure 1(a)).

These spectral limitations are compounded in oriented MOD because detection variables do not all obey the same transformation law under input rotation. Existing MOD pipelines rely on increased model capacity or rotation augmentation to learn rotational behavior implicitly, without explicitly constraining how spectral features or individual detector outputs should transform under rotation. Spatial equivariance alone is insufficient. Equivariant MOD must preserve local band order, adapt spectral–context fusion across scales, and assign rotation laws to heterogeneous outputs. This formulation reconciles invariant class and canonical-geometry readouts with equivariant orientation prediction (Figures 1(b) and 1(c)).

Three coupled limitations remain. First, band reduction, reweighting, and unconstrained spatial–spectral mixing do not preserve sensor-defined band order or its derived latent topology. Second, level-agnostic fusion cannot adjust local differential spectral evidence and high-level object context as resolution and semantic depth change. Third, current MOD pipelines lack variable-level rotation constraints that reconcile invariant and equivariant outputs within one prediction framework under response-specific transformation laws.

To address these limitations, we propose S<sup>3</sup>E-Det, a strict spatial–spectral equivariant detector for MOD. This design ties spectrally aware sampling and multi-scale aggregation to the same equivariant feature space. Specifically, Spectral Orderaware Sampling Calibration (SOSC) preserves the physical order of the input bands and constrains the input-derived latent channel topology within each block through local orderaware sampling and first-order variation. The resulting representations are propagated across scales through Common-Differential Equivariant Fusion (CDEF), which balances spectral evidence and object context according to pyramid resolution and semantic depth.

To satisfy these response-specific geometric requirements, S<sup>3</sup>E-Det adopts a lightweight Geometric Alignment Head (GAH), which decodes detection variables from shared groupindexed feature representations through response-specific readouts. GAH couples invariant category estimation with equivariant localization in a single prediction stage. The three modules preserve spectral structure and equivariant propagation through compact, response-aligned outputs.

Experiments on MODA, HOD3K, and DroneVehicle evaluate the same detector across complementary multispectral configurations and scene conditions. Across these settings, S<sup>3</sup>E-Det combines strong accuracy, stable rotation response, and a compact parameter footprint. Figure 2 summarizes the mAP –parameter trade-off on MODA. The main contributions are:

• We propose S<sup>3</sup>E-Det, a strict spatial–spectral equivariant detector that couples ordered spectral structure, groupindexed multi-scale propagation, and geometry-aligned prediction in a unified framework.

![](images/2693b04d022bd78fa5e4e950814ac9bd5a5e2fec3e46e80d3ac0f52c0a985049.jpg)  
Fig. 1: (a) Conventional spatial–spectral modeling does not preserve the ordered spectral structure and uses fixed feature fusion. (b) Ordered adaptive modeling preserves spectral order and adapts fusion across feature levels. (c) Strict rotational response consistency makes feature responses transform predictably under exact quarter-turn rotations.

• We introduce Spectral Order-aware Sampling Calibration (SOSC), which combines block-local latent spectral cells with first-order variation under order-preserving calibration during encoding.

• We develop Common-Differential Equivariant Fusion (CDEF) for scale-adaptive common–differential updates, paired with a compact Geometric Alignment Head (GAH) for response-specific invariant and equivariant readouts.

![](images/1b0faedb193c1b19d8bb70dd454e087297c68044761947c2adfeb3ded81c0dba.jpg)  
Fig. 2: $\mathrm { \ m A P _ { 5 0 } }$ vs. Params on MODA; bubble area: FLOPs.

## II. RELATED WORK

## A. Spectral Image Analysis

Multispectral and hyperspectral imaging capture wavelength-resolved, material-dependent signatures beyond conventional RGB imagery [6]. Spectral image analysis spans classification [5], object detection [7], reconstruction [8], and hyperspectral–multispectral fusion [9]. Recent work also targets transferable pretraining; SpectralGPT learns generic spatial–spectral representations across downstream tasks [10]. Multispectral oriented detection adds geometry-aligned instance perception to this landscape, requiring discriminative spectral–spatial representations and transformation-consistent localization.

## B. Multispectral Object Detection

Current multispectral detectors improve spectral selection and aggregation but leave three coupled limitations. First, S<sup>2</sup>ADet selects bands and applies principal-component analysis before two-stream spectral–spatial aggregation [7], whereas OSSDet modulates, masks, and attends across spectral features [3]. These strategies improve discrimination but treat the spectral axis mainly as evidence to select, reweight, or compress without constraining local order in input-derived spectral features; SOSC instead calibrates block-local latent spectral cells. Second, their two-stream and attention-based fusion schemes aggregate complementary cues without conditioning common object context and differential spectral evidence on pyramid level; CDEF performs this adaptation within a groupindexed pyramid. Third, these published MOD designs do not assign distinct transformation laws to category scores, canonical box distances, and orientation; GAH uses invariant readouts for the first two and an equivariant orientation response. S<sup>3</sup>E-Det couples spectral order, scale adaptation, and geometric consistency in one detector.

![](images/1904a552fe453d30d689c3f81cac246b21acebfee9ba8eb9b1806c0d884f2b93.jpg)  
Fig. 3: Overall network framework of $\mathrm { S ^ { 3 } E  – D e t }$ . SOSC blocks perform staged equivariant feature extraction, CDEF conducts multi-scale fusion, and GAH decodes invariant categories and aligned oriented boxes at three scales. (a) Stem maps multispectral inputs into group-indexed feature fields; (b) CDEF separates common context from differential spectral detail before an agreement-gated residual update.

## C. Rotation-Equivariant Object Detection

Rotation-equivariant networks originate from G-CNNs [12], which define group convolutions for discrete rotations and reflections. ReDet [13], MessDet [14], and R2Det [17] extend this idea to oriented or generic object detection. Rotation equivariance is equally relevant beyond remote sensing, particularly for natural-scene objects exhibiting diverse and unconstrained orientations [17]. Although these methods improve rotational stability, their prediction stages do not enforce response-specific transformation laws, so a shared decoding rule does not preserve the equivariant feature structure. Our geometric alignment head therefore matches response readouts to the group action at low cost.

## III. METHOD

We present $\mathrm { S ^ { 3 } E  – D e t . }$ a spatial–spectral rotation-equivariant detector, as shown in Figure 3. It coordinates representation and transformation laws across the full detector. SOSC preserves order during spectral calibration, CDEF adapts common–differential fusion in the group-indexed pyramid, and GAH decodes invariant categories and transformationconsistent boxes.

## A. Preliminaries

Let $C _ { N } = \{ r _ { 0 } , \dots , r _ { N - 1 } \}$ denote a finite cyclic group of in-plane rotations. The element $r _ { k }$ rotates the image plane by $\theta _ { k } = 2 \pi k / N$ . For a multispectral image $x \in \mathbb { R } ^ { \breve { B } \times \breve { L } \times H \times W }$ with L ordered spectral bands, we write the spatial action of $r \in C _ { N }$ as $\mathcal { T } _ { r } x$ . This action rotates the image coordinates without reordering the spectral index. The default implementation uses $N = 4 ,$ , yielding $C _ { 4 } ;$ ; Table VI evaluates alternative group orders.

A mapping Φ is rotation equivariant if applying a rotation before feature extraction is equivalent to transforming the extracted feature field:

$$
\Phi ( { \mathcal T } _ { r } x ) = \rho ( r ) \Phi ( x ) , \quad r \in C _ { N } ,\tag{1}
$$

where $\rho ( r )$ is the induced action on the feature representation. The equation states that rotating the input before feature extraction is equivalent to applying the corresponding featurespace action after feature extraction. The familiar notion of invariance is recovered when this output action is the identity, $h ( \mathcal T _ { r } x ) = h ( x )$

After lifting, the feature tensor has shape $F \in { }$ $\mathbb { R } ^ { B \times C \times N \times H \times \mathsf { \breve { W } } }$ , where the third axis indexes the $C _ { N }$ orientation channels. For each spatial position $u \in \mathbb { Z } ^ { 2 }$ and orientation index $s \in C _ { N }$ , the regular group action is

$$
[ \rho ( r ) F ] ( u , s ) = F ( r ^ { - 1 } u , r ^ { - 1 } s ) , \quad u \in \mathbb { Z } ^ { 2 } , \ s \in C _ { N } .\tag{2}
$$

Thus, input rotations rotate the spatial coordinates and induce a cyclic permutation of the group-indexed channels.

## B. Spectral Order-aware Sampling Calibration

Sensor-defined order exists at the input, whereas lifting, projection, and patch merging integrate information across bands into latent spectral–spatial features [3], [7]. SOSC carries this structure into the hierarchy as a block-local topology rather than a persistent wavelength identity. It confines each response to an intra-cell coordinate and adds a first-order variation term, preventing crossing within the current latent block.

Let $X \in \mathbb { R } ^ { B \times C \times G \times H \times W }$ denote a group-indexed feature tensor, where C indexes learned spectral-feature slots and G is the number of orientation channels. Within the current SOSC block, we use the unit-spaced canonical coordinates $s _ { i } \ = \ i ,$ satisfying $s _ { 1 } < s _ { 2 } < \cdots < s _ { C }$ , and define slot i’s cell using neighboring midpoints. These coordinates describe a blocklocal latent topology rather than a one-to-one correspondence with sensor wavelengths:

$$
\ell _ { i } = \left\{ \begin{array} { l l } { s _ { 1 } , } & { i = 1 , } \\ { \frac { s _ { i - 1 } + s _ { i } } { 2 } , } & { i > 1 , } \end{array} \right. \quad r _ { i } = \left\{ \begin{array} { l l } { \frac { s _ { i } + s _ { i + 1 } } { 2 } , } & { i < C , } \\ { s _ { C } , } & { i = C . } \end{array} \right.\tag{3}
$$

These boundaries are fixed by the ordered sampling geometry rather than learned thresholds, and adjacent cells meet only at shared midpoints. SOSC predicts an intra-cell sampling coordinate using the first-order spectral derivative along the ordered channel centers:

$$
\nabla _ { s } X _ { i } = \frac { X _ { i + 1 } - X _ { i - 1 } } { s _ { i + 1 } - s _ { i - 1 } } , \quad 1 < i < C ,\tag{4}
$$

Boundary channels use one-sided differences. A lightweight spectral encoder $\phi ( \cdot )$ , shared across positions and orientations, predicts $\rho _ { i }$ from $[ X _ { i } , \nabla _ { s } X _ { i } ]$

$$
\rho _ { i } = \operatorname { t a n h } \left( \phi _ { i } ( [ X , \nabla _ { s } X ] ) \right) .\tag{5}
$$

This coefficient chooses the sampling side and distance within the local cell:

$$
q _ { i } = s _ { i } + [ \rho _ { i } ] _ { + } ( r _ { i } - s _ { i } ) - [ - \rho _ { i } ] _ { + } ( s _ { i } - \ell _ { i } ) ,\tag{6}
$$

where $[ z ] _ { + } = \operatorname* { m a x } ( z , 0 )$ . Since $q _ { i }$ is confined to its own cell and adjacent cells only meet at shared midpoints, the sampled coordinates remain ordered:

$$
q _ { 1 } \leq q _ { 2 } \leq \cdots \leq q _ { C } .\tag{7}
$$

As shown in Figure 4, an SOSC block acts as a compact spectral-coordinate calibration unit inside each backbone stage. It preserves the stage resolution and returns the calibrated response to the original feature stream.

Thus, SOSC preserves spectral ranking by construction without a hand-tuned displacement bound. It then samples the response at $q _ { i }$ by linear interpolation:

$$
\hat { X } _ { i } = ( 1 - \omega _ { i } ) X _ { \lfloor q _ { i } \rfloor } + \omega _ { i } X _ { \lceil q _ { i } \rceil } , \quad \omega _ { i } = q _ { i } - \lfloor q _ { i } \rfloor .\tag{8}
$$

The final response keeps $X _ { i }$ and adds the sampled residual and local spectral slope:

$$
Y _ { i } = X _ { i } + \gamma _ { i } ( \hat { X } _ { i } - X _ { i } ) + \eta _ { i } \nabla _ { s } X _ { i } ,\tag{9}
$$

where $\begin{array} { r } { \gamma _ { i } ~ = ~ \sigma ( g _ { i } ) } \end{array}$ and $\eta _ { i } \ = \ \operatorname { t a n h } ( h _ { i } )$ are residual and derivative gates from the same spectral encoder. The residual gate controls the interpolated correction, while the derivative gate injects directional band variation around the original response. Because these predictors are shared across positions and orientations and act only on the spectral axis, SOSC commutes with the regular group action in Eq. (2). It refines ordered spectral responses without altering the rotationequivariant transformation law of the backbone.

![](images/d8c61a23dd942d091a5dc9ea6d0af208cbaa5df3c8b54a9d49e6c3d78d5b8640.jpg)  
Fig. 4: Structure of the SOSC block. It calibrates ordered spectral responses through neighboring-support slicing, derivativeaware interpolation, residual update, and RE-FFN.

## C. Common-Differential Equivariant Fusion

Cross-branch fusion strengthens complementarity, but a shared rule cannot balance common object context and differential spectral evidence across scales. CDEF makes this balance scale dependent under the same rotation-group action. It decomposes each group-indexed feature $F _ { l } \in$ $\mathbb { R } ^ { B \times C _ { l } \times G \times H _ { l } \times W _ { l } }$ into common and differential evidence.

Figure 3(b) gives the CDEF block used in the neck. Rather than replacing the pyramid feature, CDEF estimates a gated equivariant update from common and differential evidence and adds it back to the input feature.

We first estimate a group-invariant spectral channel gate

$$
a _ { l } = \sigma \left( \mathrm { M L P } \left( \mathrm { G A P } _ { G , H , W } ( F _ { l } ) \right) \right) ,\tag{10}
$$

and use it to reweight the feature before decomposition. Let $\mathcal { A } _ { k } ( \cdot )$ denote a local spatial average with kernel size k. The two components are computed as

$$
\begin{array} { r l } & { \mathcal { F } _ { l } ^ { c } = \psi _ { c } \left( \mathcal { A } _ { k } ( F _ { l } ) \odot a _ { l } \right) , } \\ & { \mathcal { F } _ { l } ^ { d } = \psi _ { d } \left( \mathcal { D } _ { k } \left( \psi _ { r } \left( \left( F _ { l } - \mathcal { A } _ { k } ( F _ { l } ) \right) \odot a _ { l } \right) \right) \right) . } \end{array}\tag{11a}
$$

(11b)

where $\psi _ { c } , \psi _ { r } , \psi _ { d }$ are pointwise equivariant projections and $\mathcal { D } _ { k }$ is an equivariant depthwise spatial operator. Pointwise projections mix channels at each group-indexed position, whereas $\mathcal { D } _ { k }$ captures within-channel local variation under the same group action. Thus, the common branch aggregates context shared by nearby responses, while the differential branch preserves residual cues for spectrally ambiguous objects.

CDEF fuses the two branches through an agreement gate. We concatenate $\mathcal { F } _ { l } ^ { c }$ and $\mathcal { F } _ { l } ^ { d }$ , project them into a compact equivariant subspace $Z _ { l } = \psi _ { f } ( [ \mathcal { F } _ { l } ^ { c } , \mathcal { F } _ { l } ^ { d } ] )$ , and compute

$$
m _ { l } = \sigma ( \psi _ { g } ( Z _ { l } ) ) \cdot \frac { 1 } { 2 } \left( 1 + \frac { \left. \bar { \mathcal { F } } _ { l } ^ { c } , \bar { \mathcal { F } } _ { l } ^ { d } \right. } { \left\| \bar { \mathcal { F } } _ { l } ^ { c } \right\| _ { 2 } \left\| \bar { \mathcal { F } } _ { l } ^ { d } \right\| _ { 2 } + \epsilon } \right) ,\tag{12}
$$

![](images/67e4d973ce11e5e91aec6a43c8fd1730703683707441165b807ae26c50586ec3.jpg)  
Fig. 5: Illustration of GAH. The head assigns invariant readouts to category scores and canonical distances, while equivariant group-phase decoding produces the aligned object angle.

TABLE I: Comparison with representative methods on MODA. † indicates methods originally designed for multispectral object detection.
<table><tr><td></td><td>Methods</td><td>Car</td><td>Bus</td><td>Van</td><td>Awn. bike</td><td>Truck</td><td>Tricycle</td><td>Bicycle</td><td>Pedestrian|</td><td> $\mathrm { \ m A P _ { 5 0 } }$ </td><td>mAP75 ↑</td><td>mAP↑|</td><td>#F↓</td><td>#P↓</td></tr><tr><td rowspan="4">twos-tage</td><td>LSKNet [19]</td><td>90.4</td><td>88.5</td><td>74.5</td><td>72.5</td><td>66.0</td><td>52.5</td><td>41.0</td><td>28.2</td><td>64.2</td><td>42.8</td><td>39.4</td><td>221.9G</td><td>32.6M</td></tr><tr><td>O-RCNN [18]</td><td>90.5</td><td>89.7</td><td>74.6</td><td>72.5</td><td>66.6</td><td>54.7</td><td>45.1</td><td>30.4</td><td>65.5</td><td>44.0</td><td>40.9</td><td>244.7G</td><td>55.3M</td></tr><tr><td>StripRCNN [20]</td><td>90.5</td><td>89.0</td><td>76.1</td><td>73.6</td><td>67.1</td><td>56.5</td><td>45.2</td><td>30.7</td><td>66.1</td><td>44.0</td><td>41.0</td><td>231.8G</td><td>45.2M</td></tr><tr><td>ReDet [13]</td><td>90.6</td><td>89.6</td><td>76.8</td><td>72.9</td><td>69.6</td><td>52.4</td><td>41.5</td><td>41.0</td><td>66.8</td><td>44.1</td><td>41.3</td><td>245.2G</td><td>31.6M</td></tr><tr><td rowspan="10">one-taãe</td><td>R3Det-KLD [22]</td><td>90.4</td><td>88.7</td><td>74.8</td><td>66.6</td><td>60.9</td><td>42.6</td><td>29.8</td><td>35.5</td><td>61.2</td><td>39.4</td><td>36.8</td><td>306.7G</td><td>39.6M</td></tr><tr><td>S2ANet [21]</td><td>90.4</td><td>88.7</td><td>74.4</td><td>69.1</td><td>62.7</td><td>48.9</td><td>30.1</td><td>40.7</td><td>63.1</td><td>38.7</td><td>36.7</td><td>216.5G</td><td>38.8M</td></tr><tr><td>YOLOv8n [29]</td><td>95.8</td><td>85.2</td><td>63.7</td><td>67.4</td><td>33.6</td><td>35.9</td><td>49.6</td><td>65.1</td><td>62.0</td><td>48.0</td><td>43.3</td><td>12.6G</td><td>3.2M</td></tr><tr><td>YOLOv12m [31]</td><td>96.3</td><td>85.3</td><td>68.8</td><td>74.3</td><td>40.3</td><td>36.3</td><td>57.4</td><td>69.5</td><td>66.0</td><td>52.4</td><td>46.8</td><td>115.0G</td><td>24.7M</td></tr><tr><td>O-RepPts [23]</td><td>90.5</td><td>89.2</td><td>77.7</td><td>71.2</td><td>66.2</td><td>53.1</td><td>43.0</td><td>41.1</td><td>66.5</td><td>44.1</td><td>40.9</td><td>213.6G</td><td>36.9M</td></tr><tr><td>YOLO11m [30]</td><td>96.4</td><td>88.0</td><td>68.2</td><td>75.7</td><td>44.4</td><td>38.0</td><td>58.1</td><td>68.7</td><td>67.2</td><td>54.6</td><td>48.9</td><td>126.2G</td><td>25.1M</td></tr><tr><td>DCFL [24]</td><td>90.6</td><td>89.7</td><td>77.5</td><td>71.3</td><td>67.2</td><td>55.5</td><td>44.2</td><td>44.0</td><td>67.5</td><td>44.8</td><td>41.5</td><td>241.0G</td><td>37.0M</td></tr><tr><td>MessDet [14]</td><td>90.0</td><td>89.1</td><td>75.4</td><td>71.4</td><td>70.2</td><td>54.9</td><td>48.4</td><td>45.5</td><td>68.1</td><td>42.4</td><td>40.1</td><td>616.0G</td><td>15.1M</td></tr><tr><td>OSSDet† [3]</td><td>90.5</td><td>89.9</td><td>79.2</td><td>72.7</td><td>69.7</td><td>58.8</td><td>45.3</td><td>45.7</td><td>69.0</td><td>45.9</td><td>42.7</td><td>263.1G</td><td>36.5M</td></tr><tr><td>ARS-DETR [25]</td><td>90.5</td><td>89.2</td><td></td><td></td><td>64.8</td><td>53.6</td><td>43.2</td><td>39.7</td><td></td><td></td><td></td><td></td><td>44.2M</td></tr><tr><td>DETR -ased</td><td>RHINO [26]</td><td>90.7</td><td>89.1</td><td>77.2 77.3</td><td>70.6 70.8</td><td>65.6</td><td>54.0</td><td>42.9</td><td>41.0</td><td>66.1 66.4</td><td>43.4 44.0</td><td>40.5 40.8</td><td>254.0G 319.5G</td><td>50.8M</td></tr><tr><td></td><td>O-DETR [27]</td><td>90.8</td><td>89.3</td><td>78.8</td><td>72.8</td><td>68.0</td><td>56.0</td><td>46.6</td><td>40.1</td><td>67.8</td><td>45.1</td><td>41.9</td><td>492.0G</td><td>57.6M</td></tr><tr><td></td><td>S3E-Det (no aug.)</td><td>96.0</td><td>94.2</td><td></td><td>79.5</td><td>61.5</td><td>49.6</td><td>59.0</td><td></td><td>72.6</td><td></td><td>54.0</td><td>35.6G</td><td>2.4M</td></tr><tr><td></td><td>S3E-Det (standard aug.)</td><td>97.1</td><td>94.9</td><td>71.0 72.5</td><td>80.2</td><td>62.9</td><td>50.3</td><td>60.1</td><td>70.0 71.2</td><td>73.7</td><td>59.7 60.7</td><td>54.8</td><td>35.6G</td><td>2.4M</td></tr></table>

where $\bar { \mathcal { F } } _ { l } ^ { c }$ and $\hat { \mathcal { F } } _ { l } ^ { d }$ denote orientation-averaged responses. The fused output is

$$
F _ { l } ^ { \prime } = F _ { l } + \alpha \psi _ { o } ( Z _ { l } \odot m _ { l } ) ,\tag{13}
$$

where α is learnable. This output balances gated spectral channels, common object context, and differential local evidence at each pyramid level.

## D. Geometric Alignment Head

Shared detection heads overlook that category evidence and object orientation respond differently to rotation. GAH preserves this distinction: category scores and canonical box distances are invariant to cyclic orientation shifts, whereas angles follow the shift. For a pyramid feature, an equivariant stem outputs $P _ { l } \in \mathbb { R } ^ { B \times C _ { l } \times G \times \mathbf { \dot { H } } _ { l } \times W _ { l } }$ . Let $Q _ { l } = \mathrm { P o o l } _ { G } ( P _ { l } )$ As illustrated in Figure 5, separate readouts preserve the transformation structure established by the backbone and neck.

Algorithm 1 summarizes response-specific decoding across the detection levels. The invariant readout predicts category

## Algorithm 1 Geometrically aligned prediction

# Input: pyramid features {F<sub>l</sub>}<sup>5</sup> , group basis # Output: class logits {p<sub>l</sub>} and aligned boxes $\begin{array} { c } { { \{ \vartheta _ { g } \} } } \\ { { \{ B _ { l } \} } } \end{array}$   
for l ∈ {3, 4, 5} do   
P = EquivariantStem(F ), Q = GroupPool(P )   
▷ Invariant readout   
p<sub>l</sub> = ClassReadout(Q<sub>l</sub>), d<sub>l</sub> = DistanceReadout(Q<sub>l</sub>)   
▷ Equivariant phase decoding   
π = OrientationSoftmax(P )   
v = VectorSum(π , {ϑ })   
θ = Wrap(arg(v ) + AngleResidual(Q ))   
▷ Aligned box decoding   
B = DecodeOBB(d , θ )   
end for   
return {p<sub>l</sub>, B<sub>l</sub>}<sup>5</sup><sub>l=3</sub>

logits and canonical box distances from the pooled descriptor, whereas the equivariant phase branch converts group responses into aligned orientations before rotated-box decoding.

Classification and canonical box distances are read as

$$
p _ { l } = \psi _ { \mathrm { c l s } } \left( Q _ { l } \right) , \quad d _ { l } = \psi _ { \mathrm { b o x } } \left( Q _ { l } \right) ,\tag{14}
$$

where $p _ { l }$ denotes class logits and $d _ { l }$ denotes box distances in the object canonical frame.

Orientation is decoded from the group phase rather than a free scalar. Let $\vartheta _ { g } = 2 \pi g / G$ be the basis angle of channel $g .$ The angle branch forms a phase vector:

$$
\pi _ { l , g } = \mathrm { s o f t m a x } _ { g } \left( \psi _ { \theta } ( P _ { l , g } ) \right) ,\tag{15a}
$$

$$
v _ { l } = \sum _ { g = 0 } ^ { G - 1 } \pi _ { l , g } \left( \cos \vartheta _ { g } , \sin \vartheta _ { g } \right) .\tag{15b}
$$

A bounded residual refines the group phase:

$$
\theta _ { l } = \mathrm { w r a p } \left( \mathrm { a r g } ( v _ { l } ) + \frac { \pi } { G } \operatorname { t a n h } \left( \psi _ { \Delta } ( Q _ { l } ) \right) \right) .\tag{16}
$$

The canonical distances $d _ { l }$ and aligned angle $\theta _ { l }$ are then passed to the standard rotated-box decoder. Following the regular group action in Eq. (2), GAH assigns different readouts to different response laws: quantities read from $Q _ { l }$ are invariant, whereas the phase vector follows the cyclic shift of orientation channels. This keeps category and canonical geometry fixed while orientation transforms with the input.

## IV. EXPERIMENTS

## A. Experimental Setup

Datasets and Evaluation Metrics. The three benchmarks were selected for task relevance and complementary coverage rather than scale alone. MODA [3] is the primary benchmark because its eight bands span approximately 395–950 nm and its large set of oriented instances includes small objects, low illumination, and cluttered urban backgrounds; it therefore supports joint evaluation of ordered spectral modeling, multiscale fusion, and rotation-consistent localization, as well as controlled ablation and rotation analyses. HOD3K [7] increases the input to 16 bands over 470–620 nm and includes people, cars, and bikes across diverse real-world scenes, testing whether the same ordered spectral modeling and pyramid fusion remain effective when the band count and category composition change. Its natural-scene imagery also provides a complementary setting for assessing the applicability of equivariant modeling beyond aerial scenes. DroneVehicle [2] provides paired visible–infrared UAV imagery from day to night across urban roads, residential areas, and parking lots, with dense oriented annotations for five vehicle classes; it evaluates the applicability of the proposed method to paired visible– infrared inputs under varied illumination and scene conditions. Together, these real-world datasets provide complementary coverage across spectral configurations and practical MOD scenarios. We report m $\mathrm { A P _ { 5 0 } } .$ , m $\mathrm { A P _ { 7 5 } }$ , mAP, parameters, and FLOPs to compare accuracy, localization quality, model size, and computation.

Implementation Details. Experiments are conducted on two NVIDIA RTX 3090 GPUs. Unless otherwise specified, the two $\mathrm { S ^ { 3 } E }$ -Det variants are trained from scratch with AdamW, an initial learning rate of 0.01, a final learning rate of 0.0001, and a weight decay of 0.0005. Input sizes, batch sizes, and schedules are dataset-specific. Baseline values are taken from the cited studies under their published benchmark protocols and were not uniformly retrained in our codebase. The two S<sup>3</sup>E-Det rows within each dataset share the official split, spectral input, input resolution, optimizer, training schedule, and evaluation implementation; they differ only in augmentation. The original setting disables augmentation to isolate the architectural contribution, whereas the standard-augmentation variant uses translation 0.1, scale 0.5, horizontal-flip probability 0.5, and Mosaic probability 1.0; HSV and all remaining augmentation terms are disabled.

TABLE II: Model size and computational resource profile.
<table><tr><td>Model</td><td>Params (M) FLOPs (G) Size (MB)</td><td></td><td></td><td>Peak GPU Mem (GB)</td><td>Time (min/epoch)</td></tr><tr><td>YOLOv8n</td><td>3.2</td><td>12.6</td><td>5.75</td><td>14</td><td>10</td></tr><tr><td>S3E-Det  $( C _ { 4 } )$ </td><td>2.4</td><td>35.6</td><td>5.43</td><td>20</td><td>12</td></tr></table>

## B. Results on the MODA Dataset

Data and Setups. MODA [3] contains 14,041 multispectral images and 330,191 oriented instances from eight categories. Each $1 2 0 0 \times 9 0 0$ image records eight bands spanning about 395–950 nm, with low illumination, small objects, and cluttered backgrounds. We use all eight bands. Each 1200 × 900 image is padded to $1 2 1 6 \times 9 2 8 ,$ , so that both spatial dimensions are divisible by the maximum network stride of 32 without rescaling the image content. We train for 20 epochs with a batch size of 10.

Quantitative Analysis. Table I compares $\mathrm { S ^ { 3 } E }$ -Det with published general and MOD-specific detector results. The two $\mathrm { S ^ { 3 } E }$ -Det rows provide an internally controlled augmentation comparison, while the cited baseline rows retain their published training protocols. With standard augmentation, $\mathrm { S ^ { 3 } E  – D e t }$ gives the best $\mathrm { \ m A P _ { 5 0 } }$ and $\mathrm { \ m A P _ { 7 5 } }$ with the fewest parameters. Against OSSDet, the absolute differences are 4.7%, 14.8%, and 12.1% in mAP<sub>50</sub>, $\mathrm { \ m A P _ { 7 5 } } .$ , and mAP, with 93.3% fewer parameters and 86.5% fewer FLOPs. $\mathrm { S ^ { 3 } E  – D e t }$ also gives the best bus, awning-bike, bicycle, and pedestrian results on MODA. Together, the aggregate and class-wise comparisons show that the localization gain extends across multiple categories while retaining a compact model footprint. Computational Cost and Resource Profile. The default $C _ { 4 }$ model contains 2,445,776 parameters and requires 35.60G FLOPs for one 1216 × 928 MODA input. Table VII decomposes these totals by checkpoint and aligns them with conformance residuals without combining the two. Table II summarizes parameter count, FLOPs, serialized size, measured peak GPU memory, and per-epoch training time. Peak GPU memory is the maximum CUDA allocation observed during the corresponding training run. Relative to YOLOv8n, $\mathrm { S ^ { 3 } E \mathrm { - } }$ Det uses fewer parameters while retaining a compact serialized footprint. The comparison therefore distinguishes compact model storage from the higher training memory and epoch time of the current implementation.

Qualitative Analysis. Figure 6 shows more target-centered responses and fewer false alarms or misses for small, weakly illuminated, and cluttered objects. The response patterns align with the intended roles of SOSC and CDEF in spectral calibration and scale-aware fusion. Crowded scenes remain challenging, with occasional failures to separate adjacent vehicles and detect small pedestrians.

![](images/64795498dc32700cc624c1613a378480218488de934dcbb1e27b3c5307942fb7.jpg)  
Fig. 6: Visualization comparison of detection results and feature maps obtained by different methods on MODA.

![](images/9a1e7561706b169fa2ca9f73da6a2b720c2296c293411bac7413999b435043e7.jpg)  
Fig. 7: Feature responses of $\mathrm { S ^ { 3 } E  – D e t : }$ (a) input image; (b)– (d) with/without SOSC, CDEF, and GAH. Yellow/orange dashed ellipses mark enhanced target/suppressed background responses, respectively.

The with/without comparisons in Figure 7 provide a component-wise view of these response changes; yellow and orange dashed ellipses mark strengthened target responses and attenuated interference, respectively. With SOSC, activations become more concentrated on vehicles and less dispersed over road texture. With CDEF, the fused representation further consolidates target-related evidence while reducing responses in non-target regions. GAH brings the local response extent into closer agreement with object geometry and limits mixed activations from adjacent areas. Localized changes around small vehicles and residual dense-region activations indicate improved discrimination rather than complete background suppression. The three stages show progressively more selective responses across spectral encoding, cross-level fusion, and geometric readout, consistent with the quantitative gains.

TABLE III: Ablation studies on the key components of $\mathrm { S ^ { 3 } E \mathrm { - } }$ Det.
<table><tr><td>Equiv.</td><td>SOSC</td><td>CDEF</td><td>GAH |</td><td>mAP50 ↑</td><td>mAP75 ↑</td><td>mAP↑</td></tr><tr><td>X</td><td>X</td><td>X</td><td>X</td><td>62.0</td><td>48.0</td><td>43.3</td></tr><tr><td>V</td><td>X</td><td>X</td><td>X</td><td>66.8</td><td>54.1</td><td>47.7</td></tr><tr><td></td><td>√</td><td>X</td><td>×</td><td>69.6</td><td>55.5</td><td>50.0</td></tr><tr><td></td><td>X</td><td>√</td><td>X</td><td>68.7</td><td>54.9</td><td>49.4</td></tr><tr><td></td><td>X</td><td>X</td><td>√</td><td>68.3</td><td>55.2</td><td>49.2</td></tr><tr><td></td><td>V</td><td>V</td><td>X</td><td>71.3</td><td>58.3</td><td>52.3</td></tr><tr><td></td><td>V</td><td>X</td><td>V</td><td>71.2</td><td>58.0</td><td>52.0</td></tr><tr><td></td><td>×</td><td>√</td><td></td><td>70.0</td><td>56.5</td><td>50.8</td></tr><tr><td></td><td>L</td><td>√</td><td>√</td><td>72.6</td><td>59.7</td><td>54.0</td></tr></table>

## C. Ablation Studies and Analysis

To evaluate component contributions, design sensitivity, and rotation robustness, we conduct a series of ablation and analysis experiments on MODA. Unless otherwise specified, all studies follow the MODA input setting, training schedule, and evaluation protocol.

Component Ablation. Table III separates equivariance from the proposed modules. The equivariant baseline gains 4.8%, 6.1%, and 4.4% in mAP<sub>50</sub>, $\mathrm { \ n A P _ { 7 5 } } .$ , and mAP. On this baseline, SOSC, CDEF, and GAH individually improve m $\mathsf { A P } _ { 5 0 }$ by 2.8%, 1.9%, and 1.5%, while full integration improves ${ \mathrm { m A P } } _ { 5 0 } , $ $\mathrm { \ m A P _ { 7 5 } } .$ , and mAP by 5.8%, 5.6%, and 6.3%. The full model yields the largest gains, confirming the complementary roles of SOSC, CDEF, and GAH.

Hyperparameter Study on SOSC. Table IV evaluates the spectral kernel size and the two calibration gates in SOSC. A kernel size of 3 gives the best overall result, indicating that local neighboring bands provide sufficient context for intracell prediction. A kernel size of 1 lacks adjacent spectral evidence, whereas larger kernels introduce more distant bands with limited return. Removing either the residual gate $\gamma _ { i }$ or the derivative gate η<sub>i</sub> reduces performance, showing that interpolated spectral correction and first-order variation modeling contribute jointly to order-aware calibration.

Hyperparameter Study on CDEF. Table V analyzes the bottleneck reduction ratio, the spatial kernel size, and the agreement gate in CDEF. A reduction ratio of 4 with a $3 \times 3$ spatial kernel yields the best accuracy. A smaller reduction ratio brings little additional benefit, while a larger ratio weakens the fused representation. Removing the agreement gate also decreases performance, indicating that agreement-guided fusion is useful for balancing common object context and differential spectral evidence.

![](images/93ff9cc20284ec7a99b62a09bbd8864362e103044194dbdc9fcd1193a5ba718a.jpg)  
Fig. 8: Detection robustness under input rotations on MODA. m $\mathrm { { A P } _ { 5 0 } }$ is evaluated after rotating validation images from $0 ^ { \circ }$ to $3 4 5 ^ { \circ }$ in $1 5 ^ { \circ }$ steps.

TABLE IV: Hyperparameter ablation of SOSC on MODA.
<table><tr><td>Spectral kernel</td><td>Residual gate γ</td><td>Derivative gate η</td><td> $\mathrm { m A P _ { 5 0 } \uparrow }$ </td><td>mAP↑</td></tr><tr><td>1</td><td></td><td>V</td><td>71.3</td><td>52.6</td></tr><tr><td>3</td><td>V</td><td>V</td><td>72.6</td><td>54.0</td></tr><tr><td>5</td><td></td><td>V</td><td>72.4</td><td>53.8</td></tr><tr><td>7</td><td></td><td></td><td>72.2</td><td>53.5</td></tr><tr><td>3</td><td>X</td><td>V</td><td>71.4</td><td>52.8</td></tr><tr><td>3</td><td></td><td>X</td><td>71.0</td><td>52.3</td></tr></table>

TABLE V: Hyperparameter ablation of CDEF on MODA.
<table><tr><td>Reduction ratio</td><td>Spatial kernel</td><td>Agreement gate</td><td> $\mathrm { m A P _ { 5 0 } \uparrow }$ </td><td>mAP↑</td></tr><tr><td>2</td><td>3</td><td>√</td><td>72.3</td><td>53.7</td></tr><tr><td>4</td><td>3</td><td>V</td><td>72.6</td><td>54.0</td></tr><tr><td>8</td><td>3</td><td>V</td><td>71.9</td><td>53.3</td></tr><tr><td>4</td><td>5</td><td>√</td><td>72.4</td><td>53.8</td></tr><tr><td>4</td><td>3</td><td>X</td><td>71.6</td><td>52.9</td></tr></table>

TABLE VI: Ablation of rotation group order on MODA.

<table><tr><td>Group order</td><td> $\mathrm { m A P _ { 5 0 } \uparrow }$ </td><td> $\mathrm { m A P _ { 7 5 } \uparrow }$ </td><td> $\mathrm { m A P \uparrow }$ </td><td>Params↓</td><td>FLOPs↓</td></tr><tr><td></td><td></td><td>48.0</td><td>43.3</td><td>0.6M</td><td>8.5G</td></tr><tr><td>C1 C2</td><td>62.0 70.8</td><td>57.9</td><td>51.5</td><td>1.2M</td><td>17.3G</td></tr><tr><td>C₄</td><td>72.6</td><td>59.7</td><td>54.0</td><td>2.4M</td><td>35.6G</td></tr><tr><td> $C _ { 8 }$ </td><td>72.7</td><td>59.8</td><td>54.0</td><td>5.0M</td><td>73.0G</td></tr></table>

Rotation Group Order. Table VI studies how the discrete rotation group affects the accuracy–compute trade-off. Moving from $C _ { 1 }$ to $C _ { 2 }$ brings a clear gain, showing the benefit of explicit rotation modeling. $C _ { 4 }$ further improves the results and provides a favorable balance between orientation resolution and computation. Although $C _ { 8 }$ gives a marginal $\mathrm { \ m A P _ { 5 0 } }$ increase, it roughly doubles the parameters and FLOPs, making $C _ { 4 }$ the default choice.

Exact- $C _ { 4 }$ Implementation-Conformance Protocol. Rotation accuracy measures robustness but does not verify the defining commutation identity. Following layer-wise audits in MessDet [14] and the exact/inexact distinction on sampled grids [15], we compute checkpoint-wise accumulated and operator-local residuals in FP32 on one fixed validation input. Together with interpolation and prediction-set errors, these residuals distinguish deviation propagated through the feature hierarchy from that introduced by individual operators and decoding. The checkpoint residual is

![](images/98c271c9497d74a5ff1ffa1e655064cbefa84babc6a14408cb06a3a391910702.jpg)  
Fig. 9: Detection result comparison on HOD3K. Red boxes indicate the ground truth, while green boxes denote the predictions.

TABLE VII: Layer-wise cost and $\mathrm { e x a c t - } C _ { 4 }$ residuals (mean over three non-identity elements). Means are computed from unrounded residuals; all displayed values lie in [0, 1].
<table><tr><td>Operator</td><td>Params ↓ (2,445,776)</td><td>FLOPs↓ (35.60G)</td><td>Cumulative  $\bar { e } _ { C _ { 4 } } \downarrow$ </td><td>Local  $\bar { c } _ { C _ { 4 } } \downarrow$ </td></tr><tr><td>L0 Stem (group lift)</td><td>2,080</td><td>0.54G</td><td>0.000000</td><td>0.000000</td></tr><tr><td>L1 SOSC-1</td><td>10,307</td><td>2.59G</td><td>0.000023</td><td>0.000004</td></tr><tr><td>L2 Patch merging-1</td><td>2,112</td><td>0.34G</td><td>0.000034</td><td>0.000004</td></tr><tr><td>L3 SOSC-2</td><td>38,883</td><td>2.51G</td><td>0.000227</td><td>0.000004</td></tr><tr><td>L4 Patch merging-2</td><td>8,320</td><td>0.34G</td><td>0.000059</td><td>0.000001</td></tr><tr><td>L5 SOSC-3</td><td>151,331</td><td>2.46G</td><td>0.000509</td><td>0.000005</td></tr><tr><td>L6 Patch merging-3</td><td>33,024</td><td>0.34G</td><td>0.000056</td><td>0.000008</td></tr><tr><td>L7 SOSC-4</td><td>593,500</td><td>2.44G</td><td>0.000100</td><td>0.000005</td></tr><tr><td>L8 RE-SPPF</td><td>65,792</td><td>0.27G</td><td>0.001804</td><td>0.000005</td></tr><tr><td>L9 FPN upsample-1</td><td>32,768</td><td>0.13G</td><td>0.002022</td><td>0.000000</td></tr><tr><td>L11 CDEF-1</td><td>97,825</td><td>1.42G</td><td>0.001432</td><td>0.000031</td></tr><tr><td>L12 FPN upsample-2</td><td>32,768</td><td>0.54G</td><td>0.002040</td><td>0.000000</td></tr><tr><td>L14 CDEF-2 (P3)</td><td>56,353</td><td>3.27G</td><td>0.000926</td><td>0.000025</td></tr><tr><td>L15 FPN downsample-1</td><td>25,088</td><td>0.41G</td><td>0.001150</td><td>0.000005</td></tr><tr><td>L17 CDEF-3 (P4)</td><td>223,297</td><td>3.25G</td><td>0.001185</td><td>0.000003</td></tr><tr><td>L18 FPN downsample-2</td><td>99,328</td><td>0.41G</td><td>0.000905</td><td>0.000001</td></tr><tr><td>L20 CDEF-4 (P5)</td><td>389,500</td><td>1.43G</td><td>0.000929</td><td>0.000001</td></tr><tr><td>GAH-Classification readout</td><td>324,800</td><td>7.37G</td><td>0.000242</td><td>0.000001</td></tr><tr><td>GAH-Canonical DFL readout</td><td>201,200</td><td>4.29G</td><td>0.001276</td><td>0.000023</td></tr><tr><td>GAH-Phase-angle readout</td><td>57,500</td><td>1.25G</td><td>0.001233</td><td>0.000000</td></tr><tr><td>Rotated-box decoder</td><td></td><td></td><td>0.025145</td><td>0.022454</td></tr><tr><td>Confidence filtering and NMS</td><td></td><td></td><td>0.023452</td><td>0.000000</td></tr></table>

$$
{ e } _ { l } ( \theta ) = \frac { \left\| V _ { l , \theta } \odot ( \Phi _ { l } ( \mathcal { T } _ { \theta } x ) - \widetilde { \rho } _ { l } ( \theta ) \Phi _ { l } ( x ) ) \right\| _ { 2 } } { \left\| V _ { l , \theta } \odot \widetilde { \rho } _ { l } ( \theta ) \Phi _ { l } ( x ) \right\| _ { 2 } + \epsilon } ,\tag{17}
$$

where $l \in \mathcal { L }$ indexes the audited checkpoints, $V _ { l , \theta } \in$ $\{ 0 , 1 \} ^ { H _ { l } \times W _ { l } }$ is the valid-support mask, and $\epsilon > 0$ stabilizes the denominator. For an operator $m \in { \mathcal { O } }$ with input $Z ,$ the local commutator residual is

$$
c _ { m } ( \theta ) = \frac { \| V _ { m , \theta } \odot \left( m ( \widetilde { \rho } _ { \mathrm { i n } } ( \theta ) Z ) - \widetilde { \rho } _ { \mathrm { o u t } } ( \theta ) m ( Z ) ) \right\| _ { 2 } } { \| V _ { m , \theta } \odot \widetilde { \rho } _ { \mathrm { o u t } } ( \theta ) m ( Z ) \| _ { 2 } + \epsilon } .\tag{18}
$$

Here $V _ { m , \theta }$ is the corresponding output-support mask.

For $\theta \in \{ 9 0 ^ { \circ } , 1 8 0 ^ { \circ } , 2 7 0 ^ { \circ } \}$ , inputs and features are transformed according to the discrete $C _ { 4 }$ group action: spatial positions undergo the corresponding quarter turn, while orientation channels are cyclically permuted without interpolation. We summarize exact-group conformance by $e _ { \operatorname* { m a x } } ^ { C _ { 4 } } = \operatorname* { m a x } _ { l , \theta } e _ { l } ( \theta )$ Intermediate angles use bilinear spatial rotation and fractional orientation shifts only to evaluate off-grid rotations, not as evidence of exact $C _ { 4 }$ or continuous SO(2) equivariance [17].

We quantify the interpolation floor by the valid-support round-trip error

$$
e _ { \mathrm { i n t } } ( \theta ) = \frac { \| V _ { \theta } \odot ( \mathcal { T } _ { - \theta } \mathcal { T } _ { \theta } x - x ) \| _ { 2 } } { \| V _ { \theta } \odot x \| _ { 2 } + \epsilon } .\tag{19}
$$

Here $V _ { \theta }$ is the image-space valid-support mask and is propagated to every feature scale by nearest-neighbor resizing. This control is important because interpolation and subsampling are sources of sampling-related equivariance error [15], [16].

GAH outputs are evaluated under their response-specific laws. Classification and canonical DFL use Eq. (17); orientation uses $e _ { \mathrm { a n g } } = d _ { \pi } ( \hat { \theta } ( \mathcal { T } _ { \theta } x ) , \hat { \theta } ( x ) + \theta ) / ( \pi / 2 )$ , where $d _ { \pi }$ is angular distance modulo π, and $e _ { \mathrm { r a w } } = ( e _ { \mathrm { c l s } } + e _ { \mathrm { D F L } } + e _ { \mathrm { a n g } } ) / 3$ For decoded prediction sets $\begin{array} { r c l } { \mathcal { P } _ { \theta } } & { = } & { \{ ( b _ { i } , y _ { i } ) \} } \end{array}$ and $\begin{array} { r l } { \mathcal { P } _ { 0 } } & { { } = } \end{array}$ $\{ ( b _ { j } , y _ { j } ) \}$ , let ${ \mathcal { M } } _ { \theta } \subseteq { \mathcal { P } } _ { \theta } \times { \mathcal { P } } _ { 0 }$ be their ProbIoU-based classaware Hungarian matching. Their normalized discrepancy is

$$
e _ { \mathrm { s e t } } ( \theta ) = 1 - \frac { 2 \sum _ { ( i , j ) \in { \mathcal { M } _ { \theta } } } { \bf 1 } [ y _ { i } = y _ { j } ] \mathrm { I o U } _ { \mathrm { r } } ( b _ { i } , \mathcal { T } _ { \theta } b _ { j } ) } { | \mathcal { P } _ { \theta } | + | \mathcal { P } _ { 0 } | } ,\tag{20}
$$

Unmatched and class-mismatched predictions are penalized. All residuals are dimensionless and non-additive; displayed zeros denote values below reporting precision.

Exact-C Implementation Conformance. Tables VII and VIII localize numerical deviation by operator and rotation angle, respectively. For exact quarter turns, the worst checkpoint residual is 0.002053, while the classification, canonical-DFL, and phase-angle residuals remain at or below 0.001276. Patch merging, pointwise nonlinearities, and FPN resampling contribute only $\bar { 1 } 0 ^ { - 6 }$ -scale local residuals. The set discrepancy rises to 0.025145 after rotated-box decoding and remains similar after NMS (0.023452), placing the dominant implementation deviation in decoding rather than subsequent confidence filtering and NMS. Together, these results show that, for the audited sample, the feature pathway numerically follows the discrete group action, while the three GAH branches provide invariant classification and canonical-DFL responses together with an equivariant orientation response. This conclusion concerns implementation conformance on the fixed sample, not the error distribution over validation inputs.

Off-Grid Rotation Evaluation and Error Localization. Table VIII also reveals a distinct error regime outside the discrete group: the mean interpolation residual increases to 0.013508, while the raw-head and post-NMS discrepancies reach 0.346815 and 0.662675. These values quantify sensitivity to rotations that require interpolation and fractional orientation shifts; they neither contradict exact $C _ { 4 }$ conformance nor establish continuous SO(2) equivariance. Accordingly, the strict claim covers architecture-level commutation and response-specific raw predictions under exact quarter turns on compatible grids; off-grid rotations, confidence filtering, and NMS remain outside its scope.

Robustness to Input Rotations. To assess geometric consistency, we rotate MODA validation images by fixed angles. As shown in Figure 8, S<sup>3</sup>E-Det shows smaller m $\mathsf { A P } _ { 5 0 }$ fluctuations and recovers at the $C _ { 4 }$ orthogonal angles, whereas OSSDet and YOLOv8n degrade more at intermediate angles. The periodic recovery is consistent with the discrete group structure used by the features and GAH.

## D. Results on the HOD3K Dataset

Data and Setups. HOD3K [7] contains 3,242 512 × 256 images with 16 bands from 470 to 620 nm and 15,149 instances in people, cars, and bikes. It follows a split of 2,308 training, 219 validation, and 715 testing images. We use all 16 bands, axis-aligned IoU, an input resolution of 1024×512, 14 training epochs, and a total batch size of 8.

Quantitative Results. Table IX evaluates the same $\mathrm { S ^ { 3 } E } \mathrm { - }$ Det design under the 16-band HOD3K configuration, which is distinct from the eight-band MODA setting. Relative to the published OSSDet results, the standard-augmentation row differs by 1.1%, 1.0%, and 0.7% in $\mathrm { m A P _ { 5 0 } , \ m A P _ { 7 5 } }$ , and mAP, with 93.5% fewer parameters and 79.0% fewer FLOPs. Relative to the no-augmentation setting, standard augmentation improves these metrics by 0.8%, 0.7%, and 0.6% without changing the model footprint. This isolates the contribution of augmentation within S<sup>3</sup>E-Det. Together, the results show that the accuracy–compactness pattern observed on MODA is retained under the 16-band input.

Qualitative Results. As shown in Figure 9, competing detectors tend to miss small or partially occluded instances that blend into roads and buildings, and they introduce false positives in low-light clutter. Across the shown examples, $\mathrm { S ^ { 3 } E \mathrm { - } }$ Det exhibits fewer missed detections and spurious responses. Overall, the examples support more consistent detection across road, vegetation, and building backgrounds, while the remaining errors are dominated by small, occluded, or low-visibility instances.

## E. Results on the DroneVehicle Dataset

Data and Setups. DroneVehicle [2] contains 28,439 visible– infrared pairs and 953,087 oriented annotations in five vehicle categories, split into 17,990/1,469/8,980 train/validation/test pairs. We concatenate each pair as a four-channel input and train at $6 4 0 \times 5 1 2$ for 12 epochs with batch size 16; other settings follow MODA.

Quantitative Results. Table X complements the many-band experiments with paired RGB–infrared detection. Relative to the published WaveMamba results, the standard-augmentation row differs by 2.0% in $\mathrm { \ m A P _ { 5 0 } }$ and 1.6% in mAP; the corresponding differences from DMM are 4.6% and 6.3%. Unlike the RGB–infrared-specific baselines, S<sup>3</sup>E-Det retains the architecture evaluated on MODA and HOD3K. The same architecture remains competitive across the evaluated multispectral configurations and attains the best result in four of five DroneVehicle categories. Relative to no augmentation, standard augmentation improves the car, bus, truck, freight-car, and van scores by 1.3%, 0.7%, 1.2%, 0.4%, and 0.9%, respectively. Both $\mathrm { S ^ { 3 } E  – D e t }$ variants retain 2.4M parameters and require 10.9G FLOPs under the four-channel input. Together, these results extend the observed accuracy– compactness pattern from many-band inputs to paired visible– infrared detection.

TABLE VIII: Angle-resolved implementation conformance on a $3 0 ^ { \circ }$ grid. Quarter turns are exact $C _ { 4 }$ elements; the remaining angles evaluate off-grid rotations. $e _ { \mathrm { r a w } }$ averages classification, canonical DFL, and phase-angle residuals. Mean rows are computed from unrounded residuals; all displayed values lie in [0, 1].
<table><tr><td rowspan="2">Angle</td><td rowspan="2"> $e _ { \mathrm { i n t } } \downarrow$ </td><td rowspan="2">L8 backbone ↓</td><td colspan="2">P3-P5</td><td rowspan="2">Decoded OBBs↓</td><td rowspan="2">Post-NMS predictions ↓</td></tr><tr><td>mean ↓</td><td> $e _ { \mathrm { r a w } } \downarrow$ </td></tr><tr><td>30°</td><td>0.013557</td><td>0.456471</td><td>0.376044</td><td>0.348256</td><td>0.432972</td><td>0.658733</td></tr><tr><td> $6 0 ^ { \circ }$ </td><td>0.013669</td><td>0.547496</td><td>0.396743</td><td>0.359207</td><td>0.456248</td><td>0.663884</td></tr><tr><td> $9 0 ^ { \circ }$ </td><td>0.000000</td><td>0.001776</td><td>0.001008</td><td>0.000890</td><td>0.022366</td><td>0.024853</td></tr><tr><td> $1 2 0 ^ { \circ }$ </td><td>0.013473</td><td>0.467767</td><td>0.381041</td><td>0.336827</td><td>0.503154</td><td>0.768568</td></tr><tr><td> $1 5 0 ^ { \circ }$ </td><td>0.013331</td><td>0.533103</td><td>0.383906</td><td>0.342756</td><td>0.451895</td><td>0.575583</td></tr><tr><td> $1 8 0 ^ { \circ }$ </td><td>0.000000</td><td>0.001838</td><td>0.001013</td><td>0.000934</td><td>0.034079</td><td>0.033145</td></tr><tr><td> $2 1 0 ^ { \circ }$ </td><td>0.013557</td><td>0.458926</td><td>0.374532</td><td>0.348194</td><td>0.433514</td><td>0.659853</td></tr><tr><td> $2 4 0 ^ { \circ }$ </td><td>0.013669</td><td>0.545102</td><td>0.395484</td><td>0.359601</td><td>0.459182</td><td>0.671875</td></tr><tr><td>270°</td><td>0.000000</td><td>0.001796</td><td>0.001019</td><td>0.000927</td><td>0.018990</td><td>0.012359</td></tr><tr><td> $3 0 0 ^ { \circ }$   $3 3 0 ^ { \circ }$ </td><td>0.013473 0.013331</td><td>0.466919 0.532164</td><td>0.380692</td><td>0.336736</td><td>0.501414</td><td>0.771625</td></tr><tr><td></td><td></td><td></td><td>0.384026</td><td>0.342943</td><td>0.451919</td><td>0.531280</td></tr><tr><td>Exact  $C _ { 4 }$  mean</td><td>0.000000</td><td>0.001804</td><td>0.001013</td><td>0.000917</td><td>0.025145</td><td>0.023452</td></tr><tr><td>Off-grid mean</td><td>0.013508</td><td>0.500993</td><td>0.384059</td><td>0.346815</td><td>0.461287</td><td>0.662675</td></tr></table>

TABLE IX: Comparison with state-of-the-art detectors on HOD3K.
<table><tr><td>Method</td><td> $\mathrm { \ m A P _ { 5 0 } }$  ←</td><td> $\mathrm { \ m A P _ { 7 5 } }$  ←</td><td>mAP↑</td><td> ${ \mathrm { F L O P s } } \downarrow$ </td><td>Params↓</td></tr><tr><td>CO-DETR</td><td>77.9</td><td>57.3</td><td>51.7</td><td>200.7G</td><td>64.5M</td></tr><tr><td>DINO</td><td>89.1</td><td>61.6</td><td>56.3</td><td>114.7G</td><td>47.6M</td></tr><tr><td>Faster R-CNN</td><td>89.4</td><td>62.7</td><td>56.9</td><td>206.7G</td><td>41.1M</td></tr><tr><td>ATSS</td><td>91.2</td><td>63.0</td><td>57.4</td><td>110.6G</td><td>32.2M</td></tr><tr><td>TOOD</td><td>90.3</td><td>65.6</td><td>57.8</td><td>108.5G</td><td>32.1M</td></tr><tr><td>RepPoints</td><td>92.0</td><td>65.5</td><td>58.4</td><td>104.5G</td><td>36.9M</td></tr><tr><td>VFNet</td><td>92.3</td><td>66.5</td><td>59.0</td><td>104.5G</td><td>32.8M</td></tr><tr><td>C-RCNN</td><td>90.9</td><td>66.9</td><td>59.6</td><td>149.5G</td><td>69.2M</td></tr><tr><td>S2 ADet [7]</td><td>93.4</td><td></td><td>59.8</td><td>169.2G</td><td>48.6M</td></tr><tr><td>OSSDet [3]</td><td>93.4</td><td>68.8</td><td>60.9</td><td>136.2G</td><td>37.7M</td></tr><tr><td>S3E-Det (no aug.)</td><td>93.7</td><td>69.1</td><td>61.0</td><td>28.6G</td><td>2.4M</td></tr><tr><td>S3E-Det (standard aug.)</td><td>94.5</td><td>69.8</td><td>61.6</td><td>28.6G</td><td>2.4M</td></tr></table>

TABLE X: Performance comparison on DroneVehicle.
<table><tr><td>Method</td><td> $\mathrm { { C a r } }$ </td><td> $\bf { B u s }$ </td><td> $\mathrm { T r u c k }$ </td><td> $\mathrm { F r e . }$ </td><td> $\mathrm { V a n }$ </td><td> $\mathrm { \ m A P _ { 5 0 } }$  ↑</td><td>mAP↑</td></tr><tr><td>C²Former [28]</td><td>90.2</td><td>89.8</td><td>68.3</td><td>64.4</td><td>58.5</td><td>74.2</td><td>47.3</td></tr><tr><td>YOLOv8l-RGB [29]</td><td>92.5</td><td>91.7</td><td>68.8</td><td>47.8</td><td>50.2</td><td>70.2</td><td>48.6</td></tr><tr><td>YOLOv81-IR [29]</td><td>93.4</td><td>91.9</td><td>69.3</td><td>53.7</td><td>51.1</td><td>71.9</td><td>49.1</td></tr><tr><td>SLBAF-Net [32]</td><td>90.2</td><td>89.9</td><td>76.0</td><td>68.2</td><td>59.9</td><td>76.8</td><td>49.5</td></tr><tr><td>CSOM-ODAF [33]</td><td>90.1</td><td>89.8</td><td>75.6</td><td>68.2</td><td>61.8</td><td>77.1</td><td>50.1</td></tr><tr><td>CMA [34]</td><td>95.8</td><td>93.1</td><td>75.9</td><td>59.8</td><td>59.4</td><td>76.8</td><td>50.4</td></tr><tr><td>GLFNet [35]</td><td>90.3</td><td>88.0</td><td>72.7</td><td>53.6</td><td>52.6</td><td>71.4</td><td>54.8</td></tr><tr><td>DMM [36]</td><td>90.4</td><td>88.7</td><td>77.8</td><td>63.0</td><td>66.0</td><td>77.2</td><td>55.8</td></tr><tr><td>IV-YOLO [37]</td><td>97.2</td><td>94.3</td><td>65.4</td><td>63.1</td><td>53.0</td><td>74.6</td><td>56.8</td></tr><tr><td>WaveMamba [38]</td><td>95.0</td><td>90.6</td><td>80.4</td><td>68.5</td><td>64.5</td><td>79.8</td><td>60.5</td></tr><tr><td>S3E-Det (no aug.)</td><td>96.5</td><td>93.0</td><td>79.5</td><td>68.5</td><td>67.0</td><td>80.9</td><td>61.5</td></tr><tr><td>S3E-Det (standard aug.)</td><td>97.8</td><td>93.7</td><td>80.7</td><td>68.9</td><td>67.9</td><td>81.8</td><td>62.1</td></tr></table>

Qualitative Results. Figure 10 shows that competing methods miss small, overlapping, or illumination-inconsistent vehicles and produce false positives in dense traffic. S<sup>3</sup>E-Det shows fewer misses and background false positives, with predictions better aligned with the annotations. Overall, the examples support more consistent detection under illumination and density variation, while the remaining errors are concentrated around small, occluded, or adjacent vehicles whose visible boundaries overlap. Across the displayed traffic layouts, this tendency persists from sparse, well-lit roads to crowded or illuminationinconsistent scenes with closely spaced vehicles and partially

![](images/c96f11713229fee7e725859a92befabe81af3a57de83d1654bd42a08d68850bf.jpg)  
Fig. 10: Detection result comparison on DroneVehicle. Red boxes indicate the ground truth, while green boxes denote the predictions.

overlapping boundaries.

## F. On-Device Inference Evaluation

Group-order and component variants are converted to FP16 RKNN and evaluated on an Orange Pi 5 Plus (RK3588) using RKNN Runtime 2.3.2, RKNPU driver 0.9.6, three 1.0-GHz NPU cores, batch size 1, and a padded $1 2 1 6 \times 9 2 8$ input. Statistics use 30 runs after 10 warm-ups; timing covers RKNN execution and output retrieval while excluding input loading, preprocessing, rotated-box decoding, and NMS. Peak memory denotes the host process’s maximum resident set size, average board power is measured at the 12 V DC input, and energy per image is the product of mean latency and average power.

Table XI exposes two distinct cost trends. Across rotationgroup variants, latency and peak memory grow with group order: from $C _ { 1 }$ to $C _ { 8 } ,$ , latency rises from 28.2 to 181.0 ms, throughput falls from 35.5 to 5.5 FPS, and peak memory increases from 82 to 595 MiB. Average board power grows more gradually, from 5.7 to 11.2 W, so the longer execution raises energy from 0.16 to 2.03 J/image. Within the $C _ { 4 }$ component sweep, SOSC is the dominant incremental cost, adding 27.0 ms and 105 MiB to the equivariant baseline; CDEF and GAH add 15.0–17.0 ms and 53–67 MiB. The complete model operates at 89.3 ms, 11.2 FPS, 312.4 MiB, and 0.82 J/image. These measurements identify group expansion and SOSC as the principal deployment bottlenecks, while CDEF and GAH retain smaller device-level footprints.

TABLE XI: Measured FP16 RKNN deployment costs of rotation-group and component variants.
<table><tr><td>Configuration</td><td>Mean (ms/image)↓ FPS↑ Memory (MiB)↓</td><td></td><td>Peak</td><td>Avg. Board Power (W)↓</td><td>Energy (J/image)↓</td></tr><tr><td colspan="6">Rotation-group variants</td></tr><tr><td> $C _ { 1 }$ </td><td>28.2</td><td>35.5</td><td>82</td><td>5.7</td><td>0.16</td></tr><tr><td> $C _ { 2 }$ </td><td>56.0</td><td>17.9</td><td>168</td><td>8.0</td><td>0.45</td></tr><tr><td> $C _ { 4 }$ </td><td>89.3</td><td>11.2</td><td>312.4</td><td>9.2</td><td>0.82</td></tr><tr><td> $C _ { 8 }$ </td><td>181.0</td><td>5.5</td><td>595</td><td>11.2</td><td>2.03</td></tr><tr><td colspan="6">Component variants at  $C _ { 4 }$ </td></tr><tr><td>Equivariant baseline</td><td>57.0</td><td>17.5</td><td>175</td><td>7.5</td><td>0.43</td></tr><tr><td>+SOSC</td><td>84.0</td><td>11.9</td><td>280</td><td>8.6</td><td>0.72</td></tr><tr><td>+CDEF</td><td>72.0</td><td>13.9</td><td>242</td><td>8.1</td><td>0.58</td></tr><tr><td>+GAH</td><td>74.0</td><td>13.5</td><td>228</td><td>8.4</td><td>0.62</td></tr><tr><td>Full S³E-Det</td><td>89.3</td><td>11.2</td><td>312.4</td><td>9.2</td><td>0.82</td></tr></table>

## V. CONCLUSIONS

We present S<sup>3</sup>E-Det, a strict spatial–spectral equivariant detector for MOD. Unlike fusion-centered MOD detectors, S<sup>3</sup>E-Det unifies order-preserving spectral calibration, scaleadaptive equivariant fusion, and response-specific geometric decoding. Experiments on MODA, HOD3K, and DroneVehicle establish a favorable accuracy–compactness trade-off and consistent rotation robustness, while ablations isolate the complementary contributions of SOSC, CDEF, and GAH. Separate training within each benchmark leaves zero-shot crosssensor transfer, controlled sensor perturbations, and largecorpus scaling unresolved. Future work will extend evaluation across sensors, imaging conditions, and larger multispectral datasets, including wavelength-aware adaptation through spectral pretraining.

## VI. ACKNOWLEDGMENTS

This work was supported by the National Natural Science Foundation of China (62101032), Chongqing Excellent Young Scientists Fund (CSTB2025NSCQ-JQX0017), and the High-Quality Development Special Project of the Ministry of Industry and Information Technology (TC240HAJ9-35).

## REFERENCES

[1] Vali, A.; Comai, S.; Matteucci, M. Deep learning for land use and land cover classification based on hyperspectral and multispectral earth observation data: A review. Remote Sens. 2020, 12, 2495.

[2] Sun, Y.; Cao, B.; Zhu, P.; Hu, Q. Drone-based RGB-infrared crossmodality vehicle detection via uncertainty-aware learning. IEEE Trans. Circuits Syst. Video Technol. 2022, 32, 6700–6713.

[3] Han, S.; Xu, T.; Liu, P.; Li, J. MODA: The first challenging benchmark for multispectral object detection in aerial images. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Singapore, 20–27 January 2026; Volume 40, pp. 4574–4582.

[4] Sun, W.; Du, Q. Hyperspectral band selection: A review. IEEE Geosci. Remote Sens. Mag. 2019, 7, 118–139.

[5] Ma, X.; Zhang, M.; Kan, Z.; Gao, Y.; Li, W. A lightweight spatial– spectral deformable CNN for UAV hyperspectral image classification. IEEE Trans. Geosci. Remote Sens. 2025, 63, 5524213.

[6] Hong, D.; Li, C.; Yokoya, N.; Zhang, B.; Jia, X.; Plaza, A.; Gamba, P.; Benediktsson, J.A.; Chanussot, J. Hyperspectral imaging. Nat. Rev. Methods Primers 2026, 6, 19.

[7] He, X.; Tang, C.; Liu, X.; Zhang, W.; Sun, K.; Xu, J. Object detection in hyperspectral image via unified spectral–spatial feature aggregation. IEEE Trans. Geosci. Remote Sens. 2023, 61, 5521213.

[8] Cai, Y.; Lin, J.; Hu, X.; Wang, H.; Yuan, X.; Zhang, Y.; Timofte, R.; Van Gool, L. Mask-guided spectral-wise transformer for efficient hyperspectral image reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), New Orleans, LA, USA, 19–24 June 2022; pp. 17502–17511.

[9] Li, C.; Hong, D.; Zhang, B.; Chanussot, J. Im2State: Image-to-state Mamba for hyperspectral and multispectral image fusion. Inf. Fusion 2026, 134, 104385.

[10] Hong, D.; Zhang, B.; Li, X.; Li, Y.; Li, C.; Yao, J.; Yokoya, N.; Li, H.; Ghamisi, P.; Jia, X.; Plaza, A.; Gamba, P.; Benediktsson, J.A.; Chanussot, J. SpectralGPT: Spectral remote sensing foundation model. IEEE Trans. Pattern Anal. Mach. Intell. 2024, 46, 5227–5244.

[11] Dai, J.; Qi, H.; Xiong, Y.; Li, Y.; Zhang, G.; Hu, H.; Wei, Y. Deformable convolutional networks. In Proceedings of the IEEE International Conference on Computer Vision, Venice, Italy, 22–29 October 2017; pp. 764–773.

[12] Cohen, T.; Welling, M. Group equivariant convolutional networks. In Proceedings of the 33rd International Conference on Machine Learning; PMLR: New York, NY, USA, 2016; Volume 48, pp. 2990– 2999.

[13] Han, J.; Ding, J.; Xue, N.; Xia, G.-S. ReDet: A rotation-equivariant detector for aerial object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Virtual, 19– 25 June 2021; pp. 2786–2795.

[14] Wu, X.; Wang, X.; Zhu, X.; Yang, L.; Liu, J.; Hu, X. Measuring the impact of rotation equivariance on aerial object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, Honolulu, HI, USA, 19–23 October 2025; pp. 7329–7339.

[15] Edixhoven, T.; Lengyel, A.; van Gemert, J.C. Using and abusing equivariance. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops , Paris, France, 2–6 October 2023; pp. 119–128.

[16] Zhang, R. Making convolutional networks shift-invariant again. In Proceedings of the 36th International Conference on Machine Learning , Long Beach, CA, USA, 9–15 June 2019; Volume 97, pp. 7324–7334.

[17] Wu, Z.; Liu, Y.; Dong, H.; Tang, X.; Yang, J.; Jin, B.; Chen, M.; Wei, X. R2Det: Exploring relaxed rotation equivariance in 2D object detection. In Proceedings of the International Conference on Learning Representations, Singapore, 24–28 April 2025.

[18] Xie, X.; Cheng, G.; Wang, J.; Yao, X.; Han, J. Oriented R-CNN for object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, Montreal, QC, Canada, 11–17 October 2021; pp. 3520–3529.

[19] Li, Y.; Hou, Q.; Zheng, Z.; Cheng, M.-M.; Yang, J.; Li, X. LSKNet: A foundation lightweight backbone for remote sensing. Int. J. Comput. Vis. 2024, 132, 3736–3752.

[20] Yuan, X.; Zheng, Z.; Li, Y.; Liu, X.; Liu, L.; Li, X.; Hou, Q.; Cheng, M.-M. Strip R-CNN: Large strip convolution for remote sensing object detection. In Proceedings of the AAAI Conference on Artificial Intelligence, Singapore, 20–27 January 2026; Volume 40, pp. 12259– 12267.

[21] Han, J.; Ding, J.; Li, J.; Xia, G.-S. Align deep features for oriented object detection. IEEE Trans. Geosci. Remote Sens. 2022, 60, 1–11.

[22] Yang, X.; Yang, X.; Yang, J.; Ming, Q.; Wang, W.; Tian, Q.; Yan, J. Learning high-precision bounding box for rotated object detection via Kullback–Leibler divergence. In Advances in Neural Information Processing Systems; 2021; Volume 34, pp. 18381–18394.

[23] Li, W.; Chen, Y.; Hu, K.; Zhu, J. Oriented RepPoints for aerial object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, New Orleans, LA, USA, 18–24 June 2022; pp. 1829–1838.

[24] Xu, C.; Ding, J.; Wang, J.; Yang, W.; Yu, H.; Yu, L.; Xia, G.-S. Dynamic coarse-to-fine learning for oriented tiny object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Vancouver, BC, Canada, 17–24 June 2023; pp. 7318–7328.

[25] Zeng, Y.; Chen, Y.; Yang, X.; Li, Q.; Yan, J. ARS-DETR: Aspect ratio-sensitive detection transformer for aerial oriented object detection. IEEE Trans. Geosci. Remote Sens. 2024, 62, 1–15.

[26] Lee, H.; Song, M.; Koo, J.; Seo, J. Hausdorff distance matching with adaptive query denoising for rotated detection transformer. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), Tucson, AZ, USA, 28 February–4 March 2025; pp. 1872–1882.

[27] Zhao, Z.; Xue, Q.; He, Y.; Bai, Y.; Wei, X.; Gong, Y. Projecting points to axes: Oriented object detection via point-axis representation. In Proceedings ofthe European Conference on Computer Vision, Milan, Italy, 29 September–4 October 2024; pp. 161–179.

[28] Yuan, M.; Wei, X. C<sup>2</sup>Former: Calibrated and complementary transformer for RGB-infrared object detection. IEEE Trans. Geosci. Remote Sens. 2024, 62, 1–12.

[29] Jocher, G.; Chaurasia, A.; Qiu, J. Ultralytics YOLOv8, version 8.0.0. 2023. Available online: https://github.com/ultralytics/ultralytics.

[30] Jocher, G.; Qiu, J. Ultralytics YOLO11, version 11.0.0. 2024. Available online: https://github.com/ultralytics/ultralytics.

[31] Tian, Y.; Ye, Q.; Doermann, D. YOLOv12: Attention-centric real-time object detectors. arXiv 2025, arXiv:2502.12524.

[32] Cheng, X.; Geng, K.; Wang, Z.; Wang, J.; Sun, Y.; Ding, P. SLBAF-Net: Super-lightweight bimodal adaptive fusion network for UAV detection in low recognition environment. Multimed. Tools Appl. 2023, 82, 47773–47792.

[33] Chen, C.; Qi, J.; Liu, X.; Bin, K.; Fu, R.; Hu, X.; Zhong, P. Weakly misalignment-free adaptive feature alignment for UAVs-based multimodal object detection. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Seattle, WA, USA, 17–21 June 2024; pp. 26836–26845.

[34] Jiang, C.; Ren, H.; Yang, H.; Huo, H.; Zhu, P.; Yao, Z.; Li, J.; Sun, M.; Yang, S. M2FNet: Multi-modal fusion network for object detection from visible and thermal infrared images. Int. J. Appl. Earth Obs. Geoinf. 2024, 130, 103918.

[35] Kang, X.; Yin, H.; Duan, P. Global-local feature fusion network for visible-infrared vehicle detection. IEEE Geosci. Remote Sens. Lett. 2024, 21, 1–5.

[36] Zhou, M.; Li, T.; Qiao, C.; Xie, D.; Wang, G.; Ruan, N.; Mei, L.; Yang, Y.; Tao Shen, H. DMM: Disparity-guided multispectral mamba for oriented object detection in remote sensing. IEEE Trans. Geosci. Remote Sens. 2025, 63, 1–13.

[37] Tian, D.; Yan, X.; Zhou, D.; Wang, C.; Zhang, W. IV-YOLO: A lightweight dual-branch object detection network. Sensors 2024, 24, 6181.

[38] Zhu, H.; Dong, W.; Yang, L.; Li, H.; Yang, Y.; Ren, Y.; Zhu, Q.; Feng, Z.; Li, C.; Lin, S.; Wang, R.; Luo, X.; Zhang, B. WaveMamba: Wavelet-driven mamba fusion for RGB-infrared object detection. In Proceedings of the IEEE/CVF International Conference on Computer Vision, Honolulu, HI, USA, 19–23 October 2025; pp. 11219–11229.