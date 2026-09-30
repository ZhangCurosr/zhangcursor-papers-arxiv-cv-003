# MSTypography: Multi-character Semantic Typography via Balancing Word Legibility and Object Recognizability

Xinye Yang<sup>1</sup>, Xinding Zhu<sup>1</sup>, Kai Fang<sup>1</sup>, Xinyi Ren<sup>1</sup>, Mengjian Li<sup>2</sup>, Bin Cao<sup>1</sup>, Jiazhou Chen<sup>1∗</sup>

<sup>1</sup>Zhejiang University of Technology

<sup>2</sup>Zhejiang Lab

Email: cjz@zjut.edu.cn

## Abstract

Semantic typography is a design technique where the visual representation of a word conveys its semantic meaning, while maintaining its legibility. Existing digital typography methods mainly focus on single-character scenarios. They sufer from a lack of legibility constraints and insuficient local deformation when extended to multi-character words, as the intricate structures among multiple characters are hardly preserved during the typography process. In this paper, we propose a global-tolocal typography framework for multi-character scenarios. It performs mask-driven silhouette approximation at the global level, while semantic-guided refinement at the local level, with a culling step in between to improve eficiency. To preserve word legibility, we designed structural losses (including explicit collision constraints and implicit Jacobian singular value constraints) and an OCR constraint for character-level readability. To enhance the object recognizability, we leverage semantic guidance with difusion priors, which drives the character glyph toward the target concept while preserving its structural integrity. To the best of our knowledge, this is the first multi-character semantic typography method that effectively balances word legibility and object recognizability. Evaluations on five representative languages (English, Chinese, Japanese, Korean, Arabic) demonstrate superiority over SOTA methods. Codes will be open-sourced.

## Introduction

Semantic typography is a design practice that shapes the visual form of a word to reflect its underlying meaning, while still keeping the text readable. For example, turning the letter “M” in the word "Mountain" into a mountain, or the Chinese character for “water” into a wave. As shown in Fig. 2, artists have created such typographic illustrations in diferent languages (English, Chinese, Japanese, and Korean). This technique has found widespread applications in logo design, book illustration, motion graphics, and creative typography.

Despite its widespread use, creating typographic illustrations remains labor-intensive, requiring skilled artists and extensive manual refinement. Automation is therefore appealing, yet it inevitably faces a fundamental tension: preserving word legibility while achieving object recognizability. This balance becomes especially challenging with multiple characters, where each character must remain legible while collectively forming a coherent shape.

![](images/0792d564b9ecda780bc2606c6c90ee8a08d2b721975b8cfc538b09af0346d289.jpg)  
Figure 1: Results of our method with five diferent languages. From top to bottom are initial masks, intermediate results with global deformation and final typography results. From left to right are foxes in Chinese and Korean, sharks in English, bunnies in Japanese and camels in Arabic.

In the last decade, a variety of automatic methods have been proposed for generating word art from text inputs. However, most of them focus primarily on single-character cases. For instance, Word-As-Image (Iluz et al. 2023) and its variant Textured Word-As-Image (Farzaneh and Balcisoy 2025) optimize letter contours using score distillation sampling (SDS) to produce editable SVGs, while VitaGlyph (Feng et al. 2026) introduces a subject environment dual-branch difusion mechanism. These methods can balance character clarity and semantic fidelity in single-character scenarios. However, when extended to multi-character words, they exhibit three typical artifacts: partial deformation (only some characters are altered), excessive distortion (loss of legibility), and conservative deformation (loss of object recognizability). The only attempt for multiple characters is Khattat (Hussein et al. 2024). It merely searches locally for lowloss deformation regions rather than treating the whole word as a unified entity, thereby circumventing the core challenge of global semantic typography. Consequently, existing approaches still lack a principled way to produce a coherent, semantically meaningful deformation across all characters while preserving each character’s legibility.

In this paper, we propose a global-to-local typography framework that operates at the word level. Fig. 1 shows our results in five representative languages. Our framework begins with an initial layout of a multi-character word, followed by a two-level optimization with a culling step in between. In the global level, a target mask guides the glyphs to stretch freely, overcoming the issue of conservative deformation. The culling step then selects the most promising intermediate results based on their alignment with the mask, discarding poorly deformed instances to improve eficiency and provide high-quality initializations for the next level. In the local level, we introduce ControlNet for global target guidance, whose SDS loss drives the glyphs toward the target semantics, while OCR is employed to enforce character-level readability constraints, preventing excessive distortion and partial deformation. Throughout the optimization, structural losses tailored for closed Bézier curves are embedded, including explicit collision constraints and implicit Jacobian singular value constraints, to maintain glyph topology and morphological stability.

![](images/ac647d18139ecffbaf416219b7a96d07eceed30cb569f960b7bf91f7ccb52165.jpg)  
Figure 2: The artists’ work, (a) English (Osotspa Co. 1998), (b) Chinese (Zhu 2016), (c) Japanese (Hudejii 2020) and (d) Korean (Kyriazi 2021).

The main contributions of this paper are as follows:

• A global-to-local typography framework for multicharacter words, built on diferentiable vector graphics rendering and efectively balancing word legibility and object recognizability across diverse languages.

• Mask-guided layout for semantic approximation. A target mask is leveraged to guide the initial layout of glyphs and progressively refines their arrangement to achieve coherent semantic alignment as a whole.

• Local structural constraints for legibility preservation. We integrate OCR supervision with explicit geometric losses and implicit Jacobian constraints to maintain glyph integrity throughout optimization.

## Related Works

## Semantic Typography

In recent years, several studies have advanced the field of semantic typography. Word-As-Image (Iluz et al. 2023) leverages difusion priors to optimize Bézier curves, enabling glyphs to visually convey their meaning, but it optimizes a single word as a whole, lacking independent control over individual characters and spatial coordination. Khattat (Hussein et al. 2024) extends this idea to multi-character scenarios, achieving end-to-end stylization across multiple languages via large language models and an OCR loss, yet its reliance on the OCR penalty restricts the degrees of freedom of the Bézier curves, resulting in conservative deformation magnitude and insuficient semantic expression.

In addition, numerous studies has been explored for glyph stylization, special efects, animation, and domain specific applications. In the area of style generation and font design, FontCrafter (Luo et al. 2026) proposes an element-based framework; ArtGlyphDifuser (Lu et al. 2025), FontStudio (Mu et al. 2024), VitaGlyph (Feng et al. 2026), and UniCalli (Xu et al. 2025) employ cross-modal fusion, shape-adaptive difusion, dual branch difusion, and a unified framework, respectively, to achieve high-quality artistic typography, efect rendering, and handwriting style customization. For animations, Dynamic Typography (Liu et al. 2025) adds dynamic efects to text; TypeDance (Xiao et al. 2024) focuses on logo design and OBI-Designer (Zhang et al. 2026a) is dedicated to stylize oracle bone inscription. However, most of them focus on single characters and do not explicitly address spatial layout and coordinated deformation among multiple characters.

However, aforementioned progresses did not solve the limitation for the extension of multi character phrases. Although Khattat supports multiple characters, its conservative constraints (such as the OCR loss) restrict the degrees of freedom for deformation; and existing methods generally lack geometric constraints, leading to self-intersection, collapse, or loss of legibility. Therefore, a new method that explicitly combines spatial arrangement with geometric constraints is needed to construct a unified diferentiable framework that optimizes both global layout and local glyph deformation.

## Vector Graphics Generation

Difusion-based text-to-image models, when combined with spatial condition maps, enable structure-guided generation. ControlNet (Zhang, Rao, and Agrawala 2023) and its extensions (Mou et al. 2024; Zavadski, Feiden, and Rother 2024; Mo et al. 2024; Yang et al. 2025; Choi et al. 2025; Xie et al. 2026) provide efective ways to inject edge maps, depth maps, or segmentation masks into the generation process. Complementary works (Liang et al. 2025; Han et al. 2025; Zhang et al. 2026c; Xiao et al. 2025; Tan et al. 2025) also improve layout fidelity by modeling visibility order or multi-condition interactions. However, all these methods operate on pixel representations. Pixel-based generation lacks explicit constraints on geometric properties such as stroke width, topological integrity, and character-specific structure. Consequently, directly applying them to multi glyph semantic typography often leads to inter glyph interference, stroke collapse, or loss of legibility, nor can they directly produce editable vector graphics.

As an alternative, vector graphics generation based on differentiable rendering is an important technical route. This approach builds on DifVG (Li et al. 2020) to rasterize Bézier curves into images and iteratively optimizes graphical parameters using vision-language model losses or the SDS loss. From CLIPDraw (Frans, Soros, and Witkowski 2022) to VectorFusion (Jain, Xie, and Abbeel 2023) and SVGDreamer (Xing et al. 2024), this paradigm has progressively improved semantic consistency and generation diversity. However, these early methods primarily target single glyph or simple shapes, lacking explicit constraints on spatial coordination among multiple glyphs and the ability to preserve the legibility of each glyph independently.

![](images/7397e32385cf4839826d59258aa216863ff1a0a9404e6cc10ca666ad1986a6f3.jpg)  
Figure 3: The overview of our global-to-local semantic typography framework. At the global level, PCA-based initial placement, linear/Bézier deformations optimized by mask-filling, layout, and Jacobian losses. A culling step then filters candidates by concave-hull IoU. At the local level, semantic guidance for detail refinement, with OCR, collision detection, and ARAP constraints to preserve legibility and local rigidity.

To mitigate the problem of shape decomposition caused by numerous overlapping paths during optimization, NeuralSVG (Polaczek et al. 2025) and SVGDreamer++ (Xing et al. 2025) introduce implicit regularization and adaptive primitive count, respectively. DuetSVG (Zhang et al. 2026b) further proposes a dual branches collaborative generation framework to enhance topological quality and editability. Nevertheless, these methods still cannot actively avoid inter glyph overlaps nor maintain the legibility of individual characters during deformation. In this paper, we propose a global-to-local optimization, active layout, and geometric constraints to specifically address the balance problem in multi-character semantic typography.

## Methodology

Our method follows a global-to-local optimization paradigm, as illustrated in Fig. 3. In the global level, we perform overall morphological adjustments on the input vector glyphs through linear transformations and nonlinear Bézier deformations, with the common goal of making the glyphs globally approximate the target object image as closely as possible. In the local level, we employ ControlNet to guide Stable Difusion for image generation, while introducing OCR and morphological constraints to preserve glyph structures.

## Initialization

In the initialization, we first generate an image reflecting the shape of the target object from a semantic text prompt using a difusion model and then extract its main region via image segmentation as the input mask image M. We then convert the input characters into Bézier curves and carry out prelayout operations: we compute the principal orientation of the main body (i.e., the black region) of M (of size $C \times$ $H \times W )$ using PCA (Pearson 1901), arrange the characters along this orientation, and scale them to avoid collisions. Subsequently, we generate a mesh from the sampled points of the original glyph G using Delaunay triangulation (Lee and Schachter 1980), where each Bézier curve is discretized into m samples. For i-th glyph $\mathbf { G } _ { i } .$ , its outline is discretized into $N _ { i }$ samples, defined as $\mathbf { P } _ { i } = [ \mathbf { p } _ { i , 1 } , \mathbf { p } _ { i , 2 } , \ldots , \mathbf { p } _ { i , L _ { i } } ] ,$ . These samples are then divided into $C _ { i }$ line segments, denoted as $\mathbf { q } _ { i , 1 } , \mathbf { q } _ { i , 2 } , . . . \mathbf { q } _ { i , C _ { i } }$ , where each segment $\mathbf { q } _ { i , j }$ connects two consecutive samples $\left( \mathbf { p } _ { i , j } , \mathbf { p } _ { i , j + 1 } \right)$ . For closed contours, the last segment connects $\mathbf { p } _ { i , N _ { i } }$ back to $\mathbf { q } _ { i , 1 }$

## Global Mask-guided Approximation

This level performs initial typography and deformation on the input glyphs to make their overall shape conform to the target silhouette, providing a good starting point for subsequent local optimization. The raw glyphs often deviate significantly from the target shape, and adjusting only their own control points makes it dificult to achieve large pose changes. Therefore, this level combines linear and nonlinear transformations: optimizing translation and scaling for rigid global alignment, while introducing a Bézier grid with bilinear interpolation and Coons correction (Gregory 1974; Forrest 1968) to apply smooth nonlinear deformation. The optimization is primarily driven by the fill coverage of the glyphs with respect to the target mask, embedding the glyphs into the rough framework of the target object. The total loss of the global level is:

$$
\mathcal { L } _ { \mathrm { g l o b } } = \mathcal { L } _ { \mathrm { f l l } } + \lambda _ { \mathrm { o r d } } \mathcal { L } _ { \mathrm { o r d } } + \lambda _ { \mathrm { p i x } } \mathcal { L } _ { \mathrm { p i x } } + \lambda _ { \mathrm { J a c } } \mathcal { L } _ { \mathrm { J a c } } .\tag{1}
$$

For longer words (e.g., in English), we bind adjacent glyphs into joint optimization groups to reduce computational complexity and improve optimization eficiency.

Object recognizability. A common approach for semantic guidance in text-to-image generation is Score Distillation

Sampling (SDS). However, it sufers from stochastic gradient noise during optimization and is often unstable in providing semantic guidance for multi-character scenarios, causing severe fluctuations of transformation parameters and dificulty in converging to the target shape. To reduce uncertainty, SDS is not adopt in this level; instead, we directly use mask approximation: generating a binary mask from the input image, computing the mean squared error between the deformed rendered image and the mask, and additionally penalizing overflow. The overflow $\rho$ is calculated:

$$
\rho = \frac { 1 } { \vert \Omega _ { 1 } \vert } \sum _ { i \in \Omega _ { 1 } } ( 1 - X _ { i } ) ,\tag{2}
$$

where $\Omega _ { 1 } = \{ i \mid M _ { i } = 1 \}$ , and $X _ { i }$ denotes the i-th pixel of the image X. When $\Omega _ { 1 } = \mathcal { O } , \rho$ is set to 0. The loss is:

$$
\mathcal { L } _ { \mathrm { f i l l } } ( \mathbf { X } , M ) = \mathrm { M S E } ( \mathbf { X } , M ) + \frac { \rho } { 1 - \rho } .\tag{3}
$$

Word legibility. To maintain spatial coordination among glyphs, we employ an ordering loss to enforce consistent centroid directions and avoid sharp turns, along with a pixelbased layout loss that combines area uniformity and overlap penalties from rendered glyph masks to balance sizes and separate characters. The losses are:

$$
\mathcal { L } _ { \mathrm { o r d } } = \underset { i } { \mathcal { A } } \left( \left( 1 - \mathbf { u } _ { i } \cdot \mathbf { d } \right) ^ { 2 } \right) + \underset { i } { \mathcal { A } } \left( \left( 1 - \mathbf { u } _ { i } \cdot \mathbf { u } _ { i + 1 } \right) ^ { 2 } \right) ,\tag{4}
$$

$$
{ \mathcal { L } } _ { \mathrm { p i x } } = { \underset { i } { A } } \left( ( S _ { i } - { \bar { S } } ) ^ { 2 } \right) + \lambda _ { \mathrm { o v e r } } { \underset { i , j } { A } } \left( { \frac { | A _ { i } \cap A _ { j } | } { | A _ { i } \cup A _ { j } | } } \right) ,\tag{5}
$$

where $\boldsymbol { \mathcal { A } } ( \cdot )$ represents the average, $\mathbf { u } _ { i }$ denotes the unit direction vector between the i-th and $( i + 1 )$ )-th glyphs, d is the unit reference direction vector. $A _ { i }$ represents the area of the i-th glyph, $S _ { i }$ denotes the ratio between the current area of the glyph and its initial area, and $\bar { S }$ is the mean of this ratio over the current iteration.

To ensure uniform scaling of the overall glyph and prevent local scaling distortions, we propose a geometric constraint loss based on the Jacobian matrix to regularize the deformation of each triangular face. Specifically, we construct the Jacobian matrix $\mathbf { J } _ { i }$ from the vertex coordinate diferences before and after deformation for each face, and use its singular values $\sigma _ { i 1 }$ and $\sigma _ { i 2 }$ to enforce uniform scaling. However, since singular values lack directional information and cannot detect face flipping, we further introduce the determinant $| \mathbf { J } _ { i } |$ to directly penalize flips for this loss:

$$
\mathcal { L } _ { \mathrm { J a c } } = \underset { i \in [ 1 , N _ { f } ] } { \mathcal { A } } \left( ( \sigma _ { i 1 } - \sigma _ { i 2 } ) ^ { 2 } + \lambda _ { \mathrm { H i p } } \mathrm { R e L U } ^ { 2 } ( - | \mathbf { J } _ { i } | ) \right)\tag{6}
$$

where $N _ { f }$ denotes the total number of triangular faces.

Under the constraints imposed by these loss functions, our method is able to achieve satisfactory typography results.

## Culling Step

Due to the stochastic nature of difusion-based mask generation, globally deformed glyphs do not always align well with the target masks. We attempted to improve the masks using adaptive control strategies such as SmartControl (Liu et al. 2024), but found their efect limited. Moreover, global generation is significantly faster than the subsequent local optimization, creating a computational asymmetry. To address these issues, we introduce a culling step after the global level. We therefore rapidly synthesize a large pool of global candidates and retain only the most promising ones for the expensive local refinement. Concretely, we compute the concave hull of each deformed glyph, fill its interior to obtain a binary image, and measure the IoU with the target mask. Concave hull is adopted because human perception prioritizes global silhouette and contour over internal details when judging shape conformity. All candidates are then ranked by IoU, and the top-N (N = 15) are selected as inputs for the local level.

Concave hull computation is non-diferentiable and therefore cannot be directly optimized in the global level. During global deformation, we instead approximate the alignment using the diferentiable fill loss ${ \mathcal { L } } _ { \mathrm { f i l l } }$ , which provides stable gradients for iterative optimization. In the subsequent culling step, we replace this proxy with the more accurate concave-hull IoU to filter out low-quality results ofline. This complementary design allows us to combine iterative optimization feasibility with precise candidate selection.

## Local Semantic Deformation

The primary objective of the local level is to refine details. In this level, we introduce ControlNet. It can produce more details compared to the filling algorithm employed in the global level. We use the mask image and the text prompt that were utilized for reference optimization in the global level as inputs to ControlNet, and leverage ControlNet to guide Stable Difusion, compute SDS, and generate the final image. The total loss of the local level is:

$$
\mathcal { L } _ { \mathrm { l o c } } = \mathcal { L } _ { \mathrm { S D S } } + \lambda _ { \mathrm { O C R } } \mathcal { L } _ { \mathrm { O C R } } + \lambda _ { \mathrm { c o l l } } \mathcal { L } _ { \mathrm { c o l l } } + \lambda _ { \mathrm { A R A P } } \mathcal { L } _ { \mathrm { A R A P } } .\tag{7}
$$

Object recognizability. Before computing the SDS loss, we apply random data augmentation (including slight scaling, translation, and rotation) to the rendered image to enhance the robustness of optimization, reduce the impact of SDS gradient noise, and improve the semantic consistency of the final graphic generated under spatial transformations. Then, we feed both $I _ { m a s k }$ and the augmented image into ControlNet and Stable Difusion to compute the SDS loss under the given spatial conditions. The semantic guidance loss $\nabla _ { \boldsymbol { \theta } } \mathcal { L } _ { \mathrm { S D S } }$ is:

$$
\mathbb { E } _ { t , \epsilon } \left( w ( t ) \big ( \hat { \epsilon } ( x _ { t } ; c _ { \mathrm { t e x t } } , c _ { \mathrm { c t r l } } , t ) - \epsilon \big ) \frac { \partial \mathcal { R } ( \theta ) } { \partial \theta } \right) ,\tag{8}
$$

$$
\hat { \epsilon } = \left( 1 - \gamma \right) \epsilon _ { A } ( x _ { t } ; \mathcal { D } , c _ { \mathrm { c t r l } } , t ) + \gamma \epsilon _ { A } ( x _ { t } ; c _ { \mathrm { t e x t } } , c _ { \mathrm { c t r l } } , t ) .\tag{9}
$$

![](images/80691fd00f891d6ae6491d0e88f580498e3535f680a26dae6c403f33a5aa9221.jpg)  
Figure 4: Comparison with SOTA methods (WAI for Word-As-Image, DT for Dynamic Typography, OBI for OBI-Designer, NB-sp for Neural B-splines, GPT for GPT-Image-1.5) across 5 languages (English, Chinese, Japanese, Korean, Arabic).

Word legibility. To efectively prevent originally connected or adjacent components of a glyph (e.g., the dot and the main vertical stroke in the letter “i”) from becoming completely detached or excessively separated, we use the encoder of SuryaOCR (Paruchuri and Team 2025) to extract the last layer features of each glyph after the global level. We then compare these features with those of the corresponding individual glyphs in the iteratively updated image, and use the resulting diference as a constraint:

$$
\mathcal { L } _ { \mathrm { O C R } } ( \mathbf { X } ) = \underset { i } { \mathcal { A } } \left( \mathrm { M S E } \left( \mathrm { O C R } ( X _ { i } ) , \mathrm { O C R } ( X _ { i } ^ { \mathrm { r e f } } ) \right) \right) .\tag{10}
$$

Stroke overlap severely compromises character legibility, but neither OCR nor low pass filter can fundamentally eliminate such overlap because they focus only on pixel or feature similarity and lack explicit constraints on spatial separation. Therefore, we introduce a diferentiable collision detection based on the signed distance field (SDF) (Osher and Sethian 1988; Macklin et al. 2020) at the geometric level, categorizing collisions into two types: self-intersection within a stroke and inter-stroke collisions. For self-intersection, we apply a diferentiable distance computation between Bézier segments, penalizing those whose distance falls below a threshold. The formula is:

$$
\mathcal { L } _ { \mathrm { s e l f } } = \underset { j , k \in C _ { i } } { \mathcal { A } } \left( \mathrm { R e L U } ^ { 2 } \left( 1 - \frac { d _ { \mathrm { m i n } } ( \mathbf { q } _ { i , j } , \mathbf { q } _ { i , k } ) } { \tau } \right) \right) ,\tag{11}
$$

where $d _ { \operatorname* { m i n } } \bigl ( \mathbf { q } _ { i , j } , \mathbf { q } _ { i , k } \bigr )$ is the diferentiable minimum distance based on the SDF (0 when intersecting), and τ is the distance threshold. The calculation of the inter-stroke collision ${ \mathcal { L } } _ { \mathrm { { i n t e r } } }$ is similar, but operates on curve pairs of diferent strokes. The overall collision loss is then given by:

$$
\mathcal { L } _ { \mathrm { c o l l } } = \mathcal { L } _ { \mathrm { s e l f } } + \lambda _ { \mathrm { i n t e r } } \mathcal { L } _ { \mathrm { i n t e r } } .\tag{12}
$$

To maintain the geometric morphology of the final graphic, we construct the Jacobian matrix for each triangular face in the same manner as described above, and further introduce the as-rigid-as-possible (ARAP) (Sorkine, Alexa et al. 2007) to drive the deformation towards pure rotation, thus achieving local rigidity. Specifically, we obtain the rotation matrix $\mathbf { R } _ { i }$ via polar decomposition of the Jacobian matrix and penalize the norm of the diference between $\mathbf { J } _ { i }$ and $\mathbf { R } _ { i } .$ Unlike ${ \mathcal { L } } _ { \mathrm { J a c } }$ in the global level, which primarily controls anisotropic scaling and prevents flipping, the ARAP focuses on preserving the rigidity of local shapes. This brings two benefits: first, it maintains morphological stability, preventing misalignment at stroke intersections (e.g., the crossing of the two strokes in $ { ^ { 6 6 } } \mathrm { X } ^ {  { 7 } } )$ caused by unconstrained motion of control points; second, it allows strokes to move approximately as a whole. The loss function is:

Table 1: Quantitative evaluation of CLIP error ↓ (left) and OCR error $\downarrow ( \times 1 0 ^ { - 3 } ,$ , right).
<table><tr><td>Lang.</td><td>Ours</td><td>WAI</td><td>DT</td><td>OBI</td><td>NB</td></tr><tr><td></td><td>EN 0.737±0.026</td><td>0.754±0.028</td><td> $0 . 7 7 3 ^ { \pm 0 . 0 1 1 }$ </td><td>0.757±0.0210.741±0.026</td></tr><tr><td>ZH</td><td>0.745±0.023</td><td> $0 . 7 7 0 ^ { \pm 0 . 0 2 4 }$ </td><td> $0 . 7 6 8 ^ { \pm 0 . 0 0 8 }$   $0 . 7 7 0 ^ { \pm 0 . 0 1 6 }$ </td><td> $0 . 7 6 4 ^ { \pm 0 . 0 2 0 }$ </td></tr><tr><td>JA</td><td></td><td>0.743±0.023 0.759±0.0310.769±0.009</td><td> $0 . 7 6 4 ^ { \pm 0 . 0 1 8 }$ </td><td> $0 . 7 6 8 ^ { \pm 0 . 0 1 6 }$ </td></tr><tr><td>KO</td><td></td><td>0.742±0.026 0.758±0.022</td><td> $0 . 7 6 9 ^ { \pm 0 . 0 0 8 }$   $0 . 7 6 8 ^ { \pm 0 . 0 2 0 }$ </td><td> $0 . 7 6 8 ^ { \pm 0 . 0 2 0 }$ </td></tr><tr><td>AR</td><td></td><td>0.742±0.0270.771±0.014</td><td> $0 . 7 5 3 ^ { \pm 0 . 0 1 6 }$   $0 . 7 6 4 ^ { \pm 0 . 0 1 7 }$ </td><td> $0 . 7 5 7 ^ { \pm 0 . 0 2 1 }$ </td></tr><tr><td>AVE</td><td> $\mathbf { 0 . 7 4 2 ^ { \pm 0 . 0 2 5 } }$ </td><td> $0 . 7 6 3 ^ { \pm 0 . 0 2 6 }$ </td><td> $0 . 7 6 6 ^ { \pm 0 . 0 1 3 }$   $0 . 7 6 4 ^ { \pm 0 . 0 1 9 }$ </td><td> $0 . 7 6 0 ^ { \pm 0 . 0 2 3 }$ </td></tr></table>

$$
\mathcal { L } _ { \mathrm { A R A P } } = \underset { i \in [ 1 , N _ { f } ] } { A } \left( \Vert \mathbf { J } _ { i } - \mathbf { R } _ { i } \Vert _ { F } ^ { 2 } \right) .\tag{13}
$$

In summary, the local level builds upon the deformation from the global level, introducing semantic guidance and geometric constraints to accomplish the global-to-local optimization. The two levels work synergistically to ultimately generate vector graphics that meet the requirements of both legibility and object recognizability.

## Implementation Details

The mask image M can be obtained from Stable Difusion with region segmentation, hand drawing, or existing images. In our framework, we adopt batch generation using Stable Difusion XL and segmenting by SAM (Ravi et al. 2025).

Since the OCR recognition dificulty varies across different language scripts, we independently adjusted the hyperparameters of the OCR loss for each language. In our experiments, the $\lambda _ { \mathrm { O C R } }$ is set to 0.2 for Chinese characters and Korean, while set to higher values for other languages (ranging from 0.3 to 0.6). Other hyperparameters are as follows: $\bar { \lambda _ { \mathrm { { o r d } } } } = 2 . 0 , \lambda _ { \mathrm { { p i x } } } = 0 . 0 0 2 , \dot { \lambda _ { \mathrm { { J a c } } } } = 0 . 0 2 , \lambda _ { \mathrm { { o v e r } } } = 3 0 0 0$ $\lambda _ { \mathrm { f l i p } } = 1 0 0 , \lambda _ { \mathrm { c o l l } } = \mathrm { \dot { 1 } } , \lambda _ { \mathrm { A R A P } } = 0 . 5 , \lambda _ { \mathrm { i n t e r } } = 1$

## Experiments and Discussions

All experiments were conducted on a single NVIDIA RTX 3090 GPU with 24 GB of VRAM. During training, both the global and local levels ran for 500 iterations each, and the total processing time for a single image is less than ten minutes. We selected words with the same semantics in five languages, each of which contains multiple test entries covering 2-5 characters. The input glyphs are based on common fonts: HobeauxRococeaux-Sherman for English, SimHei for Chinese, Meiryo for Japanese, Malgun Gothic for Korean, and Arial for Arabic. Vector outlines are extracted from these fonts as initial shapes. For each of the 30 test entries (6 words per language across 5 languages), our method generated 15 results per entry. For comparison, Word-As-Image (Iluz et al. 2023), Dynamic Typography (Liu et al. 2025), OBI-Designer (Zhang et al. 2026a), and Neural B-splines (Berio et al. 2025) produced 10–20 results per entry, while GPT-Image-1.5 (OpenAI 2025) generated one result per entry.

<table><tr><td>Lang.</td><td>Ours</td><td>WAI</td><td>DT</td><td>OBI</td><td>NB</td></tr><tr><td>EN</td><td></td><td>6.097±1.8389.607±5.468</td><td>16.05±4.875 7.086±3.092</td><td></td><td> $1 3 . 4 1 ^ { \pm 5 . 5 2 0 }$ </td></tr><tr><td>ZH</td><td> ${ \bf 6 . 2 1 5 ^ { \pm 1 . 7 3 4 } }$ </td><td> $6 . 8 5 3 ^ { \pm 1 . 9 9 4 }$ </td><td>7.634±2.374</td><td> $6 . 6 7 6 ^ { \pm 2 . 4 4 8 }$ </td><td> $8 . 4 6 1 ^ { \pm 3 . 1 4 5 }$ </td></tr><tr><td>JA</td><td> $5 . 7 5 9 ^ { \pm 2 . 2 4 2 }$ </td><td> $4 . 2 2 9 ^ { \pm 1 . 5 7 9 }$ </td><td> $5 . 9 1 5 ^ { \pm 1 . 9 8 8 }$ </td><td> $\mathbf { 4 . 1 7 9 ^ { \pm 1 . 9 8 5 } }$ </td><td> $6 . 0 1 7 ^ { \pm 1 . 8 5 8 }$ </td></tr><tr><td>KO</td><td> $6 . 9 2 3 ^ { \pm 1 . 8 9 7 }$ </td><td> $5 . 7 4 4 ^ { \pm 1 . 4 7 0 }$ </td><td> $6 . 4 5 3 ^ { \pm 3 . 3 3 4 }$ </td><td> $3 . 9 6 4 ^ { \pm 1 . 2 3 8 }$ </td><td> $5 . 4 2 7 ^ { \pm 1 . 8 6 7 }$ </td></tr><tr><td>AR</td><td> $4 . 4 7 1 ^ { \pm 1 . 4 7 5 }$ </td><td> $4 . 1 3 7 ^ { \pm 1 . 4 9 2 }$ </td><td> $9 . 6 4 8 ^ { \pm 2 . 7 4 1 }$ </td><td> $\mathbf { 3 . 9 7 1 ^ { \pm 1 . 9 7 8 } }$ </td><td> $5 . 2 0 1 ^ { \pm 2 . 6 8 3 }$ </td></tr><tr><td>AVE</td><td> $5 . 8 9 3 ^ { \pm 2 . 0 2 1 }$ </td><td> $6 . 1 1 9 ^ { \pm 3 . 5 1 5 }$ </td><td> $9 . 1 3 9 ^ { \pm 4 . 8 9 6 }$ </td><td> ${ \bf 5 . 1 1 7 ^ { \pm 2 . 6 1 1 } }$ </td><td> $7 . 7 2 9 ^ { \pm 4 . 5 2 2 }$ </td></tr></table>

## Qualitative Evaluation

Fig. 4 shows the experimental results. It can be observed that methods such as Word-As-Image, Dynamic Typography, and OBI-Designer tend to apply only limited deformation when handling multi-characters, or produce results where the original glyphs become unrecognizable after deformation. Neural B-Spline, on the other hand, primarily fills the target shape with little regard for preserving glyph structure. GPT-Image-1.5 essentially performs no glyph deformation and instead generates the target image by adding auxiliary forms. In contrast, our method achieves a better balance between object recognizability and word legibility in multi-character scenarios.

## Quantitative Evaluation

To objectively measure the balance between legibility and recognizability, we adopt two metrics: CLIP error for object recognizability and OCR feature error for word legibility. In our experiments, we observed that severely distorted characters led to extremely low OCR recognition accuracy, with most methods failed to correctly recognize any character. As a result, traditional recognition rate metrics were no longer applicable, and instead we adopted encoding feature diferences to measure legibility. CLIP error is computed as the cosine distance between the concave hull region of the generated image and the target text description $( \mathrm { e . g . , \tilde { \Omega } a t i g e r ^ { \ast } } )$ . The concave hull helps filter out stroke detail interference and focuses on the overall shape. For legibility, we used TrOCR (Li et al. 2023) to compute the cosine distance between the encoded features of the deformed characters and those of the initial layout images, and using the initial layout as a baseline reduces positional and layout bias.

Table 1 summarizes the quantitative results. Our method achieves the lowest CLIP distance across all five languages, with an average of0.741, demonstrating consistently superior object recognizability. For OCR feature distance, our method attains the best results on English and Chinese and ranks second on average $( 5 . 8 6 9 \times 1 0 ^ { - 3 } )$ , only behind OBI-Designer $( 5 . 0 0 3 \times 1 0 ^ { - 3 } )$ . The slightly better average OCR score of OBI-Designer stems from its inherently conservative deformation strategy, which better preserves character identity but sacrifices object recognizability, as evidenced by its substantially higher CLIP distance (averaging 0.765). In contrast, our global-to-local framework achieves a clearly more favorable trade-of, significantly improving recognizability while maintaining competitive legibility across all languages.

![](images/e0ffda338682bff4a78765590ec49dbd98d8c3f7e05958e9499e172f7d699ff2.jpg)  
Figure 5: Ablation study. (a)–(h) Global-level ablation: top row after global deformation, bottom row after full pipeline optimization. (i)–(n) Local-level ablation, showing final results.

## Ablation Study

We perform ablation experiments to examine the contribution of each design component. The results are presented in Fig. 5, with hyperparameter ablation provided in the supplementary material.

For the global level, we evaluate several variants: replacing the Fill loss with CLIP (b) or SDS+ControlNet (c) to test the influence of stochastic noise on layout arrangement; removing the Bézier grid (d) or the linear transformation (e) to assess their contribution to deformation capacity; individually ablating the arrangement losses (f) and the Jacobian loss (g); and skipping the global stage entirely while directly optimizing glyphs with SDS+ControlNet (h).

For the local level, we substitute the SDS loss with CLIP (i), remove ControlNet (j), or replace with the Fill loss (k) to verify the role of semantic guidance; remove ARAP (l) and ablate the OCR loss (m) or collision loss (n) to examine their importance for glyph integrity and structural stability.

## User Study

We recruited 22 participants with computer graphics backgrounds to rate 150 results produced by 5 methods across 5 languages and 6 semantics using a 5-point Likert scale (Likert 1932) (1 = very poor, 5 = excellent) on three criteria: word legibility, object recognizability, and overall quality.

Fig. 6 reports the mean scores and standard deviations. Our method achieves the highest ratings on word legibility (3.72) and overall quality (3.55). For object recognizability, it scores 3.68, trailing only Word-As-Image (3.80). However, Word-As-Image’s legibility is much lower (2.96), indicating that it trades of legibility for recognizability. In contrast, our global-to-local optimization yields a more favorable balance, as evidenced by the top overall quality score.

## Limitations and Future Works

Despite being the first multi-character typography method balancing legibility and recognizability, it remains sensitive to complex masks or simple glyphs, assumes a single connected mask, requires language-specific tuning, and incurs high optimization cost.We plan to extend to multi-component masks via graph-based layout decomposition, reduce mask sensitivity with self-refinement, automate parameter tuning via adaptive learning, accelerate optimization with progressive rendering, and support dynamic/interactive typography.

![](images/52aae4296f51b18b91e45142b23ab597622f923ecb287ceffb03ad70a5ace3b7.jpg)  
Figure 6: User study results: mean scores and standard devi ations across five methods and three criteria.

## Conclusion

In this paper, we propose a multi-character semantic typography framework, MSTypography. To the best of our knowledge, it is the first semantic typography method tailored for multi-character words. To achieve the best balance between the word legibility and the object recognizability efectively, structural losses and an OCR constraint for character-level readability are designed, while semantic guidance with difusion priors are introduced. Instead of local deformation, our method deforms the whole word in a two-level mechanism: it performs mask-driven silhouette approximation at the global level, while semantic-guided refinement at the local level. A number of experiments on various languages demonstrate that the proposed method outperforms SOTA methods.

## References

Berio, D.; Stroh, M.; Calinon, S.; Leymarie, F. F.; Deussen, O.; and Shamir, A. 2025. Neural Image Abstraction using Long Smoothing B-Splines. ACM Transactions on Graphics (SIGGRAPH Asia 2025 Conference Proceedings), 44(6): Accepted.

Choi, H.; Kasahara, I.; Engin, S.; Graule, M. A.; Chavan-Dafle, N.; and Isler, V. 2025. Finecontrolnet: Fine-level text control for image generation with spatially aligned text control injection. In 2025 IEEE/CVF Winter Conference on Applications ofComputer Vision (WACV), 3975–3984. IEEE.

Farzaneh, M. J.; and Balcisoy, S. 2025. Textured Word-As-Image illustration. arXiv preprint arXiv:2512.01648.

Feng, K.; Zhang, Y.; Yu, H.; Ji, Z.; Bai, J.; Zhang, H.; and Zuo, W. 2026. Vitaglyph: Vitalizing artistic typography with flexible dual-branch difusion models. In Proceedings ofthe IEEE/CVF Winter Conference on Applications of Computer Vision, 8220–8230.

Forrest, A. R. 1968. Curves and surfaces for computer-aided design. (No Title).

Frans, K.; Soros, L.; and Witkowski, O. 2022. Clipdraw: Exploring text-to-drawing synthesis through language-image encoders. Advances in Neural Information Processing Systems, 35: 5207–5218.

Gregory, J. A. 1974. Smooth interpolation without twist constraints. In Computer aided geometric design, 71–87. Elsevier.

Han, W.; Lee, Y.; Kim, C.; Park, K.; and Hwang, S. J. 2025. Spatial transport optimization by repositioning attention map for training-free text-to-image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 18401–18410.

Hudejii. 2020. Calligraphy Works of Japanese Character "Ne" for Year of the Rat New Year Card.

Hussein, A.; Elsetohy, A.; Hadhoud, S.; Bakr, T.; Rohaim, Y.; and AlKhamissi, B. 2024. Khattat: Enhancing readability and concept representation of semantic typography. In European Conference on Computer Vision, 278–295. Springer.

Iluz, S.; Vinker, Y.; Hertz, A.; Berio, D.; Cohen-Or, D.; and Shamir, A. 2023. Word-as-image for semantic typography. ACM Transactions on Graphics (TOG), 42(4): 1–11.

Jain, A.; Xie, A.; and Abbeel, P. 2023. Vectorfusion: Textto-svg by abstracting pixel-based difusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1911–1920.

Kyriazi, A. 2021. Artist uses Hangeul letters to draw endangered animal species. Artwork created by Jin Gwan-woo (Soomtangeutdeul).

Lee, D.-T.; and Schachter, B. J. 1980. Two algorithms for constructing a Delaunay triangulation. International Journal ofComputer and Information Sciences, 9(3): 219–242.

Li, M.; Lv, T.; Chen, J.; Cui, L.; Lu, Y.; Florencio, D.; Zhang, C.; Li, Z.; and Wei, F. 2023. TrOCR: transformer-based optical character recognition with pre-trained models. In Proceedings ofthe Thirty-Seventh AAAI Conference on Artificial Intelligence and Thirty-Fifth Conference on Innovative

Applications of Artificial Intelligence and Thirteenth Symposium on Educational Advances in Artificial Intelligence, 13094–13102.

Li, T.-M.; Lukáč, M.; Michaël, G.; and Ragan-Kelley, J. 2020. Diferentiable Vector Graphics Rasterization for Editing and Learning. ACM Trans. Graph. (Proc. SIGGRAPH Asia), 39(6): 193:1–193:15.

Liang, D.; Jia, J.; Liu, Y.; Ke, Z.; Fu, H.; and Lau, R. W. 2025. Vodif: Controlling object visibility order in text-toimage generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, 18379–18389.

Likert, R. 1932. A technique for the measurement of attitudes. Archives of psychology.

Liu, X.; Wei, Y.; Liu, M.; Lin, X.; Ren, P.; Xie, X.; and Zuo, W. 2024. Smartcontrol: Enhancing controlnet for handling rough visual conditions. In European Conference on Computer Vision, 1–17. Springer.

Liu, Z.; Meng, Y.; Ouyang, H.; Yu, Y.; Zhao, B.; Cohen-Or, D.; and Qu, H. 2025. Dynamic typography: Bringing text to life via video difusion prior. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 14787–14797.

Lu, X.; Chen, Y.; Rong, Y.; and Xiong, S. 2025. ArtGlyphDiffuser: Text-driven artistic glyph generation via Style-to-CLIP Projection and Multi-Level Controlled difusion. Pattern Recognition, 112172.

Luo, W.; Tan, C.; Ge, C.; Hong, B.; Yang, S.; and Ma, Y. 2026. FontCrafter: High-Fidelity Element-Driven Artistic Font Creation with Visual In-Context Generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 583–593.

Macklin, M.; Erleben, K.; Müller, M.; Chentanez, N.; Jeschke, S.; and Corse, Z. 2020. Local optimization for robust signed distance field collision. Proceedings of the ACM on Computer Graphics and Interactive Techniques, 3(1): 1–17.

Mo, S.; Mu, F.; Lin, K. H.; Liu, Y.; Guan, B.; Li, Y.; and Zhou, B. 2024. Freecontrol: Training-free spatial control of any text-to-image difusion model with any condition. In Proceedings ofthe IEEE/CVF conference on computer vision and pattern recognition, 7465–7475.

Mou, C.; Wang, X.; Xie, L.; Wu, Y.; Zhang, J.; Qi, Z.; and Shan, Y. 2024. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image difusion models. In Proceedings of the AAAI conference on artificial intelligence, volume 38, 4296–4304.

Mu, X.; Chen, L.; Chen, B.; Gu, S.; Bao, J.; Chen, D.; Li, J.; and Yuan, Y. 2024. Fontstudio: shape-adaptive difusion model for coherent and consistent font efect generation. In European Conference on Computer Vision, 305–322. Springer.

OpenAI. 2025. GPT Image 1.5. Accessed: 2026-07-28.

Osher, S.; and Sethian, J. A. 1988. Fronts propagating with curvature-dependent speed: Algorithms based on Hamilton-Jacobi formulations. Journal of computational physics, 79(1): 12–49.

Osotspa Co., L. 1998. Shark Energy Drink Logo. Brand logo consisting of a shark formed by the stylized word ’SHARK’.

Paruchuri, V.; and Team, D. 2025. Surya: A lightweight document OCR and analysis toolkit. https://github.com/datalabto/surya. GitHub repository.

Pearson, K. 1901. On lines and planes of closest fit to systems of points in space. The London, Edinburgh, and Dublin Philosophical Magazine and Journal ofScience, 2(11): 559– 572.

Polaczek, S.; Alaluf, Y.; Richardson, E.; Vinker, Y.; and Cohen-Or, D. 2025. Neuralsvg: An implicit representation for text-to-vector generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 15458–15468.

Ravi, N.; Gabeur, V.; Hu, Y.-T.; Hu, R.; Ryali, C.; Ma, T.; Khedr, H.; Rädle, R.; Rolland, C.; Gustafson, L.; et al. 2025. Sam 2: Segment anything in images and videos. In International Conference on Learning Representations, volume 2025, 28085–28128.

Sorkine, O.; Alexa, M.; et al. 2007. As-rigid-as-possible surface modeling. In Symposium on Geometry processing, volume 4, 109–116.

Tan, Z.; Liu, S.; Yang, X.; Xue, Q.; and Wang, X. 2025. Ominicontrol: Minimal and universal control for difusion transformer. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 14940–14950.

Xiao, S.; Wang, L.; Ma, X.; and Zeng, W. 2024. TypeDance: Creating semantic typographic logos from image through personalized generation. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, 1–18.

Xiao, S.; Wang, Y.; Zhou, J.; Yuan, H.; Xing, X.; Yan, R.; Li, C.; Wang, S.; Huang, T.; and Liu, Z. 2025. Omnigen: Unified image generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 13294–13304.

Xie, Y.; Feng, F.; Shi, R.; Wang, J.; Rui, Y.; and Geng, X. 2026. Divcontrol: Knowledge diversion for controllable image generation. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, 27108–27116.

Xing, X.; Yu, Q.; Wang, C.; Zhou, H.; Zhang, J.; and Xu, D. 2025. Svgdreamer++: Advancing editability and diversity in text-guided svg generation. IEEE transactions on pattern analysis and machine intelligence.

Xing, X.; Zhou, H.; Wang, C.; Zhang, J.; Xu, D.; and Yu, Q. 2024. Svgdreamer: Text guided svg generation with difusion model. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 4546–4555.

Xu, T.; Wang, K.; Chen, Z.; Wu, L.; Wen, T.; Chao, F.; and Chen, Y.-C. 2025. UniCalli: A Unified Difusion Framework for Column-Level Generation and Recognition of Chinese Calligraphy. arXiv preprint arXiv:2510.13745.

Yang, H.; Han, W.; Zhou, Y.; and Shen, J. 2025. Dccontrolnet: Decoupling inter-and intra-element conditions in image generation with difusion models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 19065–19074.

Zavadski, D.; Feiden, J.-F.; and Rother, C. 2024. Controlnetxs: Rethinking the control of text-to-image difusion models as feedback-control systems. In European Conference on Computer Vision, 343–362. Springer.

Zhang, J.; Deng, F.; Yuan, J.; Xu, C.; Long, G.; Li, R.; and Chen, S. 2026a. OBI designer: zero-shot oracle bone inscription artistic characters generation with multimodal style transfer. npj Heritage Science, 14(1): 152.

Zhang, L.; Rao, A.; and Agrawala, M. 2023. Adding conditional control to text-to-image difusion models. In Proceedings of the IEEE/CVF international conference on computer vision, 3836–3847.

Zhang, P.; Zhao, N.; Fisher, M.; Xu, Y.; Liao, J.; and Liu, D. 2026b. Duetsvg: Unified multimodal svg generation with internal visual guidance. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 10219–10229.

Zhang, X.; Bai, Z.; Wang, H.; and Song, Y. 2026c. Sigma: Selective-interleaved generation with multi-attribute tokens. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 38165–38175.

Zhu, R. 2016. Amazing Chinese Characters in Pictures. Accessed: 2026-06-08.