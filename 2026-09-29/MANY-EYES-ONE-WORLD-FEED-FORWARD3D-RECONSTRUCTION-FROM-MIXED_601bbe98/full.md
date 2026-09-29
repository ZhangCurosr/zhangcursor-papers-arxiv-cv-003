![](images/d484c18310344e11915ae8c5939dff5158dbdd2f27d0a38a8f5f5aa31de7a50d.jpg)  
Figure 1: MEOW reconstructs a shared 3D scene from mixed-camera images in one forward pass without supplied camera calibration or camera-type labels. Left: LENSCOPE generates mixedcamera training views from procedural scenes, checks their covisibility, and provides exact rays, metric depth, and camera poses for supervision. Yellow markers indicate the selected panorama centres. Right: reconstructions of mixed-camera tuples constructed from Stanford 2D3DS and laser-scanned panoramic captures. MEOW receives images only, while Wid3R additionally receives camera-type labels. Each point cloud is independently similarity-aligned to the ground truth for visualisation; VGGT panels are magnified 2×.

# MANY EYES, ONE WORLD: FEED-FORWARD3D RECONSTRUCTION FROM MIXED CAMERAS

Qiaoge Li<sup>1</sup> Yifan Zhan<sup>2</sup> Haijun Yang<sup>1</sup> Haiyang Liu<sup>2</sup> Yiyi Cai<sup>2</sup> Chenchi Luo<sup>1</sup> <sup>1</sup>China Mobile Communications Company Limited Research Institute <sup>2</sup>The University of Tokyo

## ABSTRACT

Real-world capture is heterogeneous: perspective, fisheye, and 360<sup>◦</sup> panoramic images can coexist within a single reconstruction task, yet most feed-forward 3D reconstruction models assume perspective imagery and a uniform input representation. Recent models handling several camera types are either informed of the camera type for each view or reconstruct one image pair at a time. No single-pass method reconstructs mixed-camera tuples containing full panoramas from images alone. We present MEOW, a feed-forward system that jointly reconstructs metric pointmaps and camera poses from one N-view tuple mixing perspective, fisheye and full-panorama images, in a single forward pass from images alone: no calibration, distortion parameters, camera-type labels or poses are supplied for any view. Our guiding design philosophy is to treat heterogeneous-camera reconstruction as a data-adaptation problem rather than an architectural redesign. MEOW retains a perspective-pretrained backbone and learns heterogeneous cameras entirely from a procedural data engine, which renders each scene across a continuous manifold of camera models with exact rays and depth, and certifies covisibility for every camera-sampled training tuple. Trained on synthetic tuples only, MEOW transfers zero-shot to real captures: on heterogeneous 2D3DS tuples it achieves 80.4 mAA@30 against 54.3 for Wid3R given the camera type of every view; on our laser-scanned mixed-camera benchmark it registers every four-view mixed tuple with 79.4 AUC@30. The data engine, benchmark, and complete evaluation pipeline will be released.

## 1 INTRODUCTION

From surveillance installations to consumer panoramas and robotic rigs, real-world capture mixes perspective, fisheye, and panoramic cameras with different projection geometries, fields of view, and aspect ratios. Reconstructing these images together requires relating observations across camera types and recovering their shared 3D geometry, often without reliable calibration. Feed-forward reconstruction models, including DUSt3R (Wang et al., 2024), VGGT (Wang et al., 2025a), π<sup>3</sup> (Wang et al., 2026), and MapAnything (Keetha et al., 2026), provide a foundation for this task, but are developed primarily for perspective imagery. Their training distributions and input processing do not adequately cover heterogeneous projections and full panoramas, limiting their performance on mixed-camera tuples (Figure 1, Table 3).

Existing approaches address parts of this problem. Rectifying wide-angle images to perspective views requires calibration and can discard field of view. Wid3R (Jung et al., 2026) jointly processes multiple camera models but requires a camera-model token for each view. CAM3R (Guruprasad et al., 2026) removes this requirement, but processes image pairs; reconstructing a larger collection requires repeated pairwise inference followed by global alignment. Extending joint reconstruction to mixed-camera tuples without supplied camera information presents two challenges. First, diverse multi-view training tuples that combine perspective, fisheye, and full-panorama images with accurate geometric supervision are scarce. Second, a shared input pipeline must accommodate their different image layouts without discarding the coverage needed for reconstruction.

We present MEOW, which adapts the publicly available MapAnything checkpoint to jointly reconstruct metric pointmaps and camera poses from tuples mixing perspective, fisheye, and fullpanorama images. Reconstruction takes one network forward pass, without externally supplied calibration, distortion parameters, camera-type labels, or poses. Our approach retains the perspectivepretrained backbone and uses synthetic data for all subsequent adaptation. LENSCOPE supplies geometrically supervised mixed-camera training tuples, while lightweight input adaptations preserve the coverage and layout information needed to process these views together.

LENSCOPE generates camera diversity from procedurally constructed indoor scenes (Section 3.2). At each sampled optical centre, we render a full panorama with exact rays, metric depth, and camera pose, then resample it into perspective, fisheye, or panoramic views with varying camera parameters. This construction varies field of view, distortion, and image layout while retaining geometric supervision. Camera sampling also changes which parts of the scene remain visible: two overlap ping panoramic observations may produce narrow views with little shared content. We therefore recompute covisibility after sampling the cameras and retain tuples whose views form a connected covisibility graph. The resulting training stream combines diverse camera projections with overlapping observations that support joint reconstruction.

To process heterogeneous views together, we resize every image to a common tensor shape without cropping (Section 3.3). This preserves the complete image coverage, including all longitudes of a full panorama. Resizing can change aspect ratios, however, so the resulting tensor shape no longer identifies the source layout. An aspect-ratio embedding supplies this information to the network, while circular padding in the dense prediction head handles the longitude boundary of full panoramas. These panoramas are identified from their images at inference, without supplied camera-type labels. We also adapt geometric supervision to the different projections: solid-angle weighting accounts for unequal directional coverage per pixel, and a directional likelihood supervises predicted rays (Section 3.4).

We evaluate MEOW on mixed-camera tuples constructed from real panoramic captures, with perspective and fisheye views resampled from the original panoramas. On heterogeneous Stanford 2D3DS tuples, MEOW reaches 80.4 mAA@30, compared with 54.3 for Wid3R supplied with camera-type labels. On our laser-scanned benchmark, where all methods are evaluated without fine-tuning, MEOW reaches 79.4 AUC@30 on mixed tuples, compared with 29.3 for Wid3R. Controlled comparisons show that synthetic adaptation substantially improves the pretrained backbone, whereas changing its input resizing alone does not. Further ablations show that full-field-of-view resizing and the aspect-ratio embedding contribute to the adapted model’s performance.

Our contributions are threefold:

• We adapt a perspective-pretrained reconstruction model to jointly recover metric pointmaps and camera poses from mixed-camera tuples containing full panoramas, in one forward pass and without externally supplied calibration, distortion parameters, camera-type labels, or poses.

• We develop LENSCOPE, a reproducible procedural data engine that generates heterogeneous camera views with exact geometric supervision and checks covisibility after camera sampling, enabling adaptation using synthetic tuples only.

• We introduce a laser-scanned benchmark with mixed-camera and single-camera tracks, and evaluate reconstruction methods without fine-tuning on it. We will release the data engine, benchmark, and evaluation pipeline.

## 2 RELATED WORK

Feed-forward reconstruction. Feed-forward reconstruction has moved from image pairs to large view collections. DUSt3R (Wang et al., 2024) predicts pointmaps from two images, and MASt3R (Leroy et al., 2024) adds matching. MUSt3R, Fast3R, and CUT3R (Cabon et al., 2025; Yang et al., 2025; Wang et al., 2025b) extend this line to many views, while VGGT and π<sup>3</sup> (Wang et al., 2025a; 2026) predict joint N-view geometry. MapAnything (Keetha et al., 2026) adds metric scale and optional geometric cues; Depth Anything 3 (Lin et al., 2026a) accommodates variable view counts. Yet these models remain centred on perspective images in a common tensor layout. Rig3R (Li et al., 2025) uses rig metadata when available and infers it otherwise, while Pow3R (Jang et al., 2025) accepts optional camera and depth priors. We build on the MapAnything backbone.

Non-pinhole cameras in feed-forward reconstruction. Existing methods differ chiefly in what camera information they require. Wid3R (Jung et al., 2026) needs a per-view camera token, which UCE (Chidlovskii et al., 2026) infers from the image. Rig3R (Li et al., 2025) can infer missing rig structure but targets perspective rigs; Fisheye3R (Duan et al., 2026) gates calibration tokens by camera type; and X-Lens (Zhou et al., 2026) takes per-pixel calibration and a camera type for every view and predicts metric depth but no poses. CAM3R (Guruprasad et al., 2026) needs no calibration, but reconstructs two views at a time and aligns them afterward. PanoVGGT, CasaMaestro, and Argus (Guo et al., 2026; Ji et al., 2026; Li et al., 2026c) handle panoramas alone, while RIGOR and HALO-SLAM (Huang et al., 2026; Xiong et al., 2026) use frozen foundation models for panoramic SLAM. Ray-aware alternatives use supplied rays for mixed pinhole/fisheye views (Zhang et al., 2026), adapt to one fisheye camera per sequence (Sinitsyn et al., 2026), or address novel-view synthesis and calibrated rig depth (Griffiths & Dansereau, 2026; Zhang et al., 2025). Among the systems in Table 1, MEOW alone combines mixed tuples containing full panoramas, no supplied calibration, distortion, or camera-type labels, one joint forward pass, and metric pointmaps with poses.

Monocular any-camera geometry and calibration. Monocular depth has expanded beyond perspective cameras (Piccinelli et al., 2025; Guo et al., 2025; Ganesan et al., 2026; Li et al., 2026b; Lin et al., 2026b; Gangopadhyay et al., 2025); OmniPoint (Ye et al., 2026) and PaGeR (Bozic et al., 2026) recover metric geometry from an arbitrary-camera image and a panorama, respectively. Another line estimates the camera model from one image (Veicht et al., 2024; Tirado-Gar´ın & Civera,

Table 1: Input requirements and outputs of the closest systems, read from their papers and code (all cited in Section 2). $\checkmark :$ supported; (✓): partial or qualitative by the authors’ own account; –: not supported or not claimed. Only MEOW takes a mixed tuple with full panoramas from pixels alone and returns metric pointmaps and poses in one pass.
<table><tr><td>Method</td><td>Camera input at test time</td><td>Mixed tuple Full 360°</td><td></td><td>N-view pass Poses Metric</td><td></td><td></td></tr><tr><td>Wid3R</td><td>Class per view</td><td>√</td><td>√</td><td>√</td><td>√</td><td></td></tr><tr><td>CAM3R</td><td>None</td><td>(√) pairs</td><td>√</td><td></td><td>√</td><td>一</td></tr><tr><td>Fisheye3R</td><td>None</td><td>(√)</td><td>(√)</td><td>√</td><td>√</td><td>一</td></tr><tr><td>X-Lens</td><td>Intrinsics or rays, and class</td><td>√</td><td>一</td><td>√</td><td>一</td><td>√</td></tr><tr><td>G-ray</td><td>Ray map per view</td><td>√</td><td></td><td>√</td><td>√</td><td>一</td></tr><tr><td>PanoVGGT</td><td>Panoramas only</td><td></td><td>√</td><td>√</td><td>√</td><td>一</td></tr><tr><td>MEOW (ours) None</td><td></td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

2025; Bogdan et al., 2018; Lochman et al., 2021) or from sparse views (Li et al., 2026a). We let multiple views constrain one another, jointly recovering geometry and poses without supplied camera parameters.

Camera models and synthetic data. Our camera manifold draws on classical models (Kannala & Brandt, 2006; Geyer & Daniilidis, 2000; Khomutenko et al., 2016; Usenko et al., 2018; Mei & Rives, 2007). Synthesising cameras from panoramas (Bogdan et al., 2018; Guruprasad et al., 2026; Jung et al., 2026) and generating procedural multi-view data (Ma et al., 2026) are established practices. Existing resources span fixed indoor scenes (Structured3D, PanoCity, OmniRooms (Zheng et al., 2020; Guo et al., 2026; Song et al., 2026)), procedural scenes (PanoInfinigen (Bozic et al., 2026)), curated asset views (CM-EVS (Liu et al., 2026)), and real sequences with laser geometry (Holo360D (Ou et al., 2026)). We generate scenes reproducibly from seeds using procedural geometry and CC0 textures on a single GPU. Our engine samples a continuous camera manifold with photographic priors over principal point, roll, tilt, and optical blur, checks covisibility for each tuple, and outputs exact rays and depth for views with distinct optical centres.

## 3 METHOD

## 3.1 PROBLEM FORMULATION

Given N images captured by unknown and potentially different camera models, our goal is to reconstruct metric pointmaps and camera poses in a single forward pass. Writing $\boldsymbol { I _ { i } } \in \mathbb { R } ^ { \breve { H } _ { i } \times W _ { i } \times 3 }$ for the i-th image, MEOW is one network $f _ { \theta }$ with

$$
f _ { \theta } \big ( \{ I _ { i } \} _ { i = 1 } ^ { N } \big ) = \Big ( \{ \mathbf { r } _ { i } , ~ d _ { i } , ~ \mathbf { R } _ { i } , ~ \mathbf { t } _ { i } \} _ { i = 1 } ^ { N } , ~ s \Big ) , \qquad \mathbf { X } _ { i } ( \mathbf { p } ) = s \big ( \mathbf { R } _ { i } d _ { i } ( \mathbf { p } ) \mathbf { r } _ { i } ( \mathbf { p } ) + \mathbf { t } _ { i } \big ) ,\tag{1}
$$

where $\mathbf { r } _ { i } ( \mathbf { p } ) \in \mathbb { S } ^ { 2 }$ is the unit ray of pixel p in the camera frame of view $i , d _ { i } ( \mathbf { p } ) > 0$ the depth along that ray, $\left( \mathbf { R } _ { i } , \mathbf { t } _ { i } \right)$ the camera-to-world pose in the frame of the first view (OpenCV convention), $s > 0$ one metric scale for the tuple and $\mathbf { X } _ { i }$ the resulting metric pointmap. At inference, no camera types, intrinsics, distortion parameters, or poses are externally supplied; the only auxiliary cues are each image’s native shape $\mathsf { \bar { ( } } H _ { i } , W _ { i } \mathsf { ) }$ and a binary panorama flag $\pi _ { i }$ inferred from its pixels. We study whether high-quality synthetic data can lift a perspective-pretrained geometry model to this mixed-camera task. Starting from the publicly available MapAnything checkpoint, we perform all subsequent adaptation on synthetic mixed-camera tuples. We first present LENSCOPE, our synthetic data engine (Section 3.2); then describe how images retain their full field of view within a shared tensor shape (Section 3.3); and finally detail the model, losses and training procedure (Section 3.4).

## 3.2 LENSCOPE: THE CAMERA-MANIFOLD ENGINE

Figure 2 draws the pipeline: an offline stage renders and indexes every scene once, and an online stage assembles one tuple per training sample. Appendix B lists the parameters and gives the complete procedure as Algorithm 1. Scenes come from our own procedural indoor generator, in the spirit of ProcTHOR (Deitke et al., 2022), in two generations: 2,004 first-generation rooms and 499 denser and more realistic second-generation scenes. Panorama poses are placed in the free space of each scene, and every pose k is rendered once as an equirectangular panorama $E _ { k }$ with analytic unit rays, radial depth and a validity mask. Panoramic sources allow fields of view up to $3 6 0 ^ { \circ }$ with exact rays and depth. Fisheye views that are warped from perspective frames, by contrast, cannot exceed the field of view of their source.

Covisibility. Two forms of one visibility test are used. Offline, between the panorama poses $k , l$ of a second-generation scene, $\operatorname { c o v } ( k  l )$ is estimated with 96 directions drawn uniformly on the sphere at pose k. Each direction is cast to the nearest solid, unambiguous surface point, which counts as covisible if the straight line from pose l reaches it unoccluded (within 5 cm). The stored value is $c _ { k l } = \operatorname* { m i n } \left( \operatorname { c o v } ( k \to \bar { l } ) , \operatorname { c o v } ( l \to k \bar { ) } \right)$ , an equal-solid-angle estimate that does not depend on the panorama grid. Online, on the sampled views, the test runs on the pixel grid of each view rendered at 96 pixels on the long side. $\mathbf { A }$ valid pixel p of view i is covisible in $j$ when its 3D point, expressed in the frame of $j$ as $\bar { \mathbf { Y } } = \mathbf { R } _ { j } ^ { \top } \left( \mathbf { X } _ { i } ( \mathbf { p } \bar { ) } - \mathbf { t } _ { j } ^ { \top } \right)$ , lies in front of $j ,$ falls within $3 . 6 ^ { \circ }$ of the nearest ray $\mathbf { q } ^ { \star } = \arg \operatorname* { m a x } _ { \mathbf { q } } \langle \mathbf { Y } / \| \mathbf { Y } \| , \mathbf { r } _ { j } ( \mathbf { q } ) \rangle$ of a valid pixel, and is not occluded there:

![](images/ce4e4c53caa6fb96756e60abb95621b90e46e7ec669d8ebdcbcdbf07ae2574d7.jpg)  
Figure 2: Offline, once per scene: a seed generates a procedural scene, panorama poses are placed in its free space, each pose is rendered with exact rays and depth; pairwise covisibility is computed by mutual visibility tests (lines: $c _ { k l } \geq 0 . 4 0 )$ . Online, per tuple: a connected walk plans 2–8 poses and each is resampled through a sampled camera. Covisibility is recomputed (edge width; dashed below 0.25) and only connected tuples are kept.

$$
| \mathbf { \| } \mathbf { Y } \| - d _ { j } ( \mathbf { q } ^ { \star } ) \big | \leq 0 . 1 0 \mathrm { m } + 0 . 0 5 d _ { j } ( \mathbf { q } ^ { \star } ) .\tag{2}
$$

cov $( i  j )$ is the fraction of valid pixels of i that pass, and the online check uses $c _ { i j } = { \textstyle { \frac { 1 } { 2 } } } ( \cos ( i \to$ $j ) + \mathrm { c o v } ( j  i ) )$ . The two directions differ by construction: a narrow view in the forward hemisphere of a panorama is covered entirely while the panorama is not. Each direction is normalised by its own sample count, so views of different resolution and field of view are comparable. Covisibility is recomputed online because a narrow or tilted camera sees a fraction of what its panorama saw.

Tuple sampling and camera manifold. A tuple of $K \in [ 2 , 8 ]$ poses with distinct optical centres is drawn by a random walk on the offline matrix that adds a pose only if its covisibility with the current pose is at least 0.40 (relaxed to 0.20 in the aimed mode below when no chain of length K exists). With probability 0.55 a tuple is planned in an aimed mode, in which the cameras converge on a common surface point (probability 0.90) or look along a shared direction toward it; otherwise each camera keeps the orientation of its panorama pose. A plan that fails is retried in the other mode. Each pose is then resampled through a camera drawn from seven models with coupled fieldof-view ranges (Table 5, Appendix B). Rectilinear OpenCV and pinhole cameras form the majority; Fisheye624, EUCM and Mei cover the fisheye family; spherical crops and full panoramas cover the sphere. Initial fields of view are log-normal around an $\mathrm { \dot { 8 } 0 ^ { \circ } }$ diagonal for the rectilinear models and log-uniform over the model’s range for the others, and distortion coefficients are drawn from ranges valid for that field of view. Full panoramas receive a random $S O ( 3 )$ content rotation. Principal-point shifts and roll follow priors set from calibration statistics and photographic collections (Clarke et al., 1998; Hold-Geoffroy et al., 2023). An optics layer (modulation transfer, vignetting, sensor noise) is decoupled from geometry, so blur is a weaker cue to the field of view.

Covisibility check and repair. The sampled views are rendered at low resolution, $c _ { i j }$ is recomputed with Eq. (2), and the tuple is kept only if the graph with edges $c _ { i j } > 0 . 2 5$ is connected. A disconnected view is moved in three steps toward a wide field of view, a square aspect and no tilt or principal-point offset, and the graph is re-tested; a tuple that still fails is discarded. Kept views are rendered once more at their target resolution. No real image of any camera type enters training after the public perspective pretraining. Scene realism matters: under the same configuration, second-generation scenes instead of first-generation renders lower Matterport3D (Chang et al., 2017) zero-shot accuracy and completeness error by 17.4% and 24.1%. Table 7 (Appendix D) contrasts this construction with the training data of the compared methods, read from their released code.

![](images/e10da5b17a59f1fd296757bb5622dac416d1f0145347ae34c96594149974ef28.jpg)  
Figure 3: One forward pass. Views from different cameras are resized to one tensor shape with their full field of view; the native aspect of each (fixed to 2:1 for detected full panoramas) enters through an MLP added to its patch tokens. Sixteen alternating global and frame attention layers, a dense head per view (circular padding for panoramas), a pose head and a scale head give rays, depth, camera-to-world poses (OpenCV axes) and metric scale in one shared world. Inputs: a real tuple, zero-shot.

## 3.3 MIXED-CAMERA INPUTS

Resizing without cropping. All views of a tuple enter the network at one tensor shape $( h , w )$ drawn from ten buckets B whose sides are multiples of 14 and whose width-to-height ratios range from 0.49 to 3.08. View i is resized anisotropically, $\tilde { I } _ { i } ( u , v ) = I _ { i } ( u W _ { i } / w , \ v H _ { i } / h )$ , so that the tensor grid covers the entire native image and nothing is cropped (Figure 7, Appendix D). Groundtruth rays, depth and masks are resampled through the same map, and rays keep their directions. During training the bucket is drawn per batch independently of the content, so the deformation $\delta _ { i } = \bar { ( } w / h ) / \bar { ( W _ { i } / H _ { i } ) }$ of each view is random. At inference the bucket closest to the mean aspect ratio of the tuple is used for all views.

Aspect-ratio embedding. Since the bucket is drawn independently of the content, the tensor shape no longer tells the network the native aspect of a view. A small MLP $g$ with a zero-initialised last layer maps $\underline { { a } } _ { i } \ : = \ : \log ( W _ { i } / H _ { i } )$ to a vector that is added to every patch token of view $i , \ \mathbf { F } _ { i } \ \gets$ $\bar { \mathbf { F } _ { i } } + \mathbf { 1 } \bar { g ( a _ { i } ) ^ { \intercal } }$ , following the global-representation encoder of MapAnything. The embedding holds 0.08% of the parameters.

Panorama flag, wrap and detection. A full equirectangular panorama is periodic in longitude. For views with $\pi _ { i } = 1$ the dense head pads its token grid with three circular columns on each side, runs, and crops back, so no parameter is added, and the aspect input takes its definitional value $a _ { i } = \log 2$ whatever the shape in which the panorama arrives; other views run the head unchanged. In training $\pi _ { i }$ is read from the ground-truth rays (azimuth span of the middle row above $3 5 0 ^ { \circ } ) ;$ at inference it comes from a detector. A full equirectangular image has two signatures that survive any aspect squeeze: its last and first columns are adjacent in content, and its top and bottom rows compress all longitudes into near-constant colour. The detector measures both on a 128-row copy of the image as ratios against its interior, with a guard against circular fisheyes with black corners; its thresholds were fixed on training renders only (Appendix C). On the two headline benchmarks it misses no panorama and flags 2 of 386 other views (Table 6, Appendix C).

## 3.4 MODEL, TRAINING AND INFERENCE

Architecture. Figure 3 presents the overall system, whose reconstruction network retains the original MapAnything backbone without architectural modification. The first 24 layers of a DI-NOv2 (Oquab et al., 2024) ViT-G encoder, shared across views, turn each resized view into patch tokens $\mathbf { F } _ { i } \in \mathbb { R } ^ { ( h w / 1 4 ^ { 2 } ) \times 1 5 3 6 }$ , to which $g ( a _ { i } )$ is added. 16 transformer layers (width 1536, 24 heads) alternate global attention over the tokens of all views together with one learnable scale token, and frame attention within each view. One DPT (Ranftl et al., 2021) dense head, applied to every view with the wrap of Section 3.3, predicts $\mathbf { r } _ { i } , d _ { i }$ , a confidence and a validity logit. One pose head per view (convolutional pooling and an MLP) predicts $\mathbf { R } _ { i }$ as a unit quaternion and $\mathbf { t } _ { i } .$ , and a scale head on the scale token predicts s. Pointmaps follow from Eq. (1).

Losses. Ground truth carries an asterisk and $m _ { i }$ is the valid mask of view i. Two changes make the MapAnything objective aware of non-perspective geometry. First, the ray, depth and point terms of view i are weighted by the solid angle of its ground-truth ray field,

$$
\omega _ { i } ( \mathbf { p } ) = \frac { \big \| \partial _ { u } \mathbf { r } _ { i } ^ { * } \times \partial _ { v } \mathbf { r } _ { i } ^ { * } \big \| ( \mathbf { p } ) } { \operatorname* { m e a n } _ { \mathbf { p } ^ { \prime } } \big \| \partial _ { u } \mathbf { r } _ { i } ^ { * } \times \partial _ { v } \mathbf { r } _ { i } ^ { * } \big \| ( \mathbf { p } ^ { \prime } ) } , \qquad \omega _ { i } \in [ 0 . 0 5 , 2 0 ] ,\tag{3}
$$

normalised to mean one over all pixels of each view, so that the poles of a panorama and the rim of a fisheye, which the pixel grid over-represents, no longer dominate; on a pinhole view $\omega _ { i }$ is proportional to cos<sup>3</sup> of the off-axis angle. Second, ray directions are scored by a von Mises–Fisher likelihood on $\mathbb { S } ^ { 2 }$ (Mardia & Jupp, 2000) with one fixed concentration $\kappa = e ^ { 3 }$

$$
\mathcal { L } _ { \mathrm { r a y } } = \sum _ { i } \sum _ { \mathbf { p } } \omega _ { i } ( \mathbf { p } ) \Big [ - \kappa \big \langle \mathbf { r } _ { i } ( \mathbf { p } ) , \mathbf { r } _ { i } ^ { * } ( \mathbf { p } ) \big \rangle + \log \frac { 4 \pi \sinh \kappa } { \kappa } \Big ] ,\tag{4}
$$

which, with κ fixed, is a cosine loss scaled by κ plus a constant. The remaining terms follow MapAnything. With $\begin{array} { r } { \rho ( \mathbf { e } ) \ = \ \frac { | \alpha - 2 | } { \alpha } \big [ \big ( \frac { \| \mathbf { e } \| ^ { 2 } / c ^ { 2 } } { | \alpha - 2 | } + 1 \big ) ^ { \alpha / 2 } \ - \ 1 \big ] } \end{array}$ the robust penalty of Barron (2019) $( \alpha = 0 . 5 , c = 0 . 0 5 )$ , applied in log space after predictions and targets are normalised by their mean distance to the origin,

$$
\begin{array} { l } { \displaystyle { \mathcal { L } = \sum _ { i } \Big [ \rho \big ( \mathbf { X } _ { i } - \mathbf { X } _ { i } ^ { * } \big ) + 0 . 1 \left( \rho ( \mathbf { P } _ { i } - \mathbf { P } _ { i } ^ { * } ) + \rho ( d _ { i } - d _ { i } ^ { * } ) + \rho ( \mathbf { q } _ { i } - \mathbf { q } _ { i } ^ { * } ) + \rho ( \mathbf { t } _ { i } - \mathbf { t } _ { i } ^ { * } ) \right) } } \\ { \displaystyle ~ } \\ { \displaystyle { ~ + ~ 0 . 3 \left( \mathcal { L } _ { \mathrm { n o r m a l } } + \mathcal { L } _ { \mathrm { g r a d } } \right) \Big ] + 0 . 1 \rho ( s - s ^ { * } ) + 0 . 1 \mathcal { L } _ { \mathrm { r a y } } + 0 . 0 3 \mathcal { L } _ { \mathrm { m a s k } } , } } \end{array}\tag{5}
$$

where dense terms are averaged over the valid pixels of each view position across the tuples of a batch (the ray term over all pixels) and the loss is scaled by $2 / K , \bar { \mathbf { P } _ { i } } = d _ { i } \mathbf { r } _ { i }$ is the camera-frame pointmap, the pose terms are applied to the absolute poses and to all pairwise relative poses (the quaternion term to the nearer of $\mathbf { q } _ { i } ^ { * }$ and −q<sup>∗</sup>), L<sub>normal</sub> compares normals of the camera-frame pointmaps, $\mathcal { L } _ { \mathrm { g r a d } }$ matches log-depth gradients, and $\mathcal { L } _ { \mathrm { m a s k } }$ is a binary cross-entropy on the validity logit. The world-point term is confidence-weighted as in DUSt3R $( c _ { \mathbf { p } } \rho - 0 . 2 \log c _ { \mathbf { p } } )$

Training and inference. Training starts from the public MapAnything checkpoint. Stage 1 finetunes the checkpoint on initial synthetic renders with the official pipeline for 35 epochs. Stage 2 turns on the full-field-of-view resizing, the aspect-ratio embedding and the online camera sampling of Section 3.2 (100 epochs). Stage 3 trains on second-generation scenes with the losses of Eqs. (3)– (5) and the panorama wrap, in two runs from Stage 2: one of 15 epochs and one of 859 epochs; the final weights interpolate the two (Appendix $\mathbf { A } )$ . Batches hold tuples of 2–8 views at the ten aspect buckets with up to 48 images per GPU on four H200 GPUs (at most 25 on eight); an epoch has 416 steps per GPU and takes 33 minutes on four H200s in Stage 2. For inference, the $\hat { N }$ images are resized to the bucket closest to their mean aspect ratio, $a _ { i }$ is read from their native shapes (set to log 2 for detected full panoramas) and $\pi _ { i }$ from the detector, and one forward pass returns Eq. (1); no alignment, matching or optimisation follows.

## 4 EXPERIMENTS

## 4.1 PROTOCOLS

Both real benchmarks start from real panoramic captures: the panoramas are the originals, and the perspective and fisheye views are resampled from them through the stated camera models, so every view carries exact rays and metric ground truth. Pose metrics are computed over ordered view pairs up to a $3 0 ^ { \circ }$ threshold: relative rotation and translation accuracy (RRA@30, RTA@30; sign-agnostic translation direction, Wang et al., 2023), their mean average accuracy (mAA@30) and area under the curve (AUC@30); ATE is the absolute trajectory error after one Sim(3) alignment per tuple (definitions in Appendix F). Every baseline runs with its official loader and checkpoint on the same tuples, ground truth and scorer; pointmap metrics and further protocol details are in Appendix F.

![](images/a90ba9d66ad5066f2648c04ea3912cf271fbf4d61cebd5124385f748ad92c7b4.jpg)  
Figure 4: Laser-scanned mixed tuples, zero-shot. Each row shows the four inputs, the laser ground truth and the fused pointmaps of MEOW (pixels only) and Wid3R (camera types given), placed by the scorer’s alignment. Top-down floor-to-wall slices coloured by height; triangles mark stations or predicted cameras.

## 4.2 HETEROGENEOUS TUPLES

Table 2 reports the heterogeneous 2D3DS benchmark: areas 5a, 5b and 6 of Stanford 2D3DS (Armeni et al., 2017), 88 tuples of 3–24 views cycling through full panoramas, synthesised 90<sup>◦</sup> perspective views and 180<sup>◦</sup> equidistant fisheyes, all views in one forward pass. MEOW reaches 80.4 mAA@30; Wid3R, given the camera class of every view, reaches 54.3, and the difference lies in translation (RTA@30 96.2 against 86.5, ATE 0.62 against 0.83). The perspective models, which also receive pixels only, reach 10.5–19.8. Input resizing alone does not explain the gain: the public MapAnything weights score 16.4 with their own loader and 15.4 with our full-field-of-view resizing (Table 8, Appendix D).

## 4.3 A LASER-SCANNED BENCHMARK ON WHICH EVERY METHOD IS ZERO-SHOT

We will release a four-track benchmark built from our own registered BLK360 G2 laser scans of a multi-room office: 12 stations with survey-grade poses and pointmaps, tuples selected by covisibility computed from the scans, and regions the scanner cannot see masked out; no method has trained on it. Table 3 gives the results and Figure 4 shows three mixed tuples. On the four-view mixed track MEOW registers every tuple (RRA@30 and RTA@30 of 100) and reaches 79.4 AUC@30 against 19.0–32.9 for the open baselines, which collapse on the panorama track. Wid3R registers the rotations of the mixed tuples but not their translations (RTA@30 58.0, AUC@30 29.3), and its pointmaps are less complete.

## 4.4 SINGLE-CAMERA INPUTS AND EFFICIENCY

On single-camera input the model behaves like its backbone on pinholes (85.7 against 86.8 AUC@30 on the laser pinhole track, Table 3) and, unlike any perspective model, registers panoramaonly tuples: 75.4 AUC@30 on 2D3DS and 71.0 on the laser track against at most 8.4 (Table 9,

Table 2: Heterogeneous 2D3DS tuples (Armeni et al., 2017): 88 tuples of 3–24 views, one forward pass. Pose metrics are computed per tuple and averaged over the 88 tuples; ATE is after Sim(3) alignment. <sup>†</sup>No multi-view weights released.
<table><tr><td>Method</td><td>Camera input</td><td>RRA@30</td><td>RTA@30</td><td>mAA@30</td><td>ATE↓</td></tr><tr><td>VGGT</td><td>None</td><td>32.6</td><td>47.3</td><td>10.5</td><td>1.65</td></tr><tr><td>73</td><td>None</td><td>45.9</td><td>59.7</td><td>19.8</td><td>1.24</td></tr><tr><td>MapAnything</td><td>None</td><td>52.8</td><td>53.9</td><td>16.4</td><td>1.48</td></tr><tr><td>CAM3R†</td><td>None</td><td></td><td></td><td></td><td></td></tr><tr><td>MEOW (ours)</td><td>None</td><td>95.2</td><td>96.2</td><td>80.4</td><td>0.62</td></tr><tr><td>Wid3R</td><td>Class per view</td><td>96.9</td><td>86.5</td><td>54.3</td><td>0.83</td></tr></table>

Table 3: Laser-scanned benchmark, 24 four-view tuples per track. AUC@30 is the mean over tuples; Acc and Comp are in metres after one alignment per tuple; NC is normal consistency.
<table><tr><td></td><td></td><td colspan="6">Mixed tuples (panorama + fisheye + two pinholes)</td><td colspan="3">Single-camera tuples, AUC@30</td></tr><tr><td>Method</td><td>Camera input</td><td>RRA@30</td><td>RTA@30</td><td>AUC@30</td><td>Acc ↓</td><td>Comp ↓</td><td>NC</td><td>Panorama</td><td>Pinhole</td><td>Fisheye</td></tr><tr><td>DUSt3R</td><td>None</td><td>62.5</td><td>67.4</td><td>32.9</td><td>0.164</td><td>0.863</td><td>0.758</td><td>0.5</td><td>96.0</td><td>73.8</td></tr><tr><td>MASt3R</td><td>None</td><td>66.7</td><td>66.0</td><td>31.8</td><td>0.182</td><td>0.508</td><td>0.737</td><td>2.7</td><td>96.2</td><td>56.2</td></tr><tr><td>VGGT</td><td>None</td><td>45.8</td><td>63.9</td><td>19.0</td><td>0.279</td><td>0.591</td><td>0.665</td><td>1.9</td><td>97.1</td><td>55.0</td></tr><tr><td>π3</td><td>None</td><td>66.0</td><td>63.9</td><td>26.6</td><td>0.240</td><td>0.693</td><td>0.736</td><td>1.6</td><td>97.9</td><td>75.1</td></tr><tr><td>MapAnything</td><td>None</td><td>66.0</td><td>57.3</td><td>24.5</td><td>0.255</td><td>0.830</td><td>0.653</td><td>1.5</td><td>86.8</td><td>75.1</td></tr><tr><td>MEOW (ours)</td><td>None</td><td>100</td><td>100</td><td>79.4</td><td>0.142</td><td>0.367</td><td>0.809</td><td>71.0</td><td>85.7</td><td>73.2</td></tr><tr><td>Wid3R</td><td>Class per view</td><td>97.2</td><td>58.0</td><td>29.3</td><td>0.199</td><td>0.467</td><td>0.764</td><td>81.9</td><td>90.5</td><td>87.3</td></tr><tr><td>PanoVGGT</td><td>Panoramas</td><td></td><td></td><td>一</td><td></td><td></td><td></td><td>90.0</td><td></td><td></td></tr></table>

Appendix D). Inference costs what MapAnything costs on the same views, 0.66 s for eight views on one RTX 5090 with image loading included, and 32 views fit in one network forward pass (Table 13, Appendix D).

## 4.5 ABLATIONS

Table 4 changes one thing at a time. On the final weights, the public crop loader in place of the fullfield-of-view resizing costs 3.3 mAA@30 on 2D3DS and 5.0 AUC@30 on the laser mixed track, and withholding the aspect-ratio embedding costs 23.2 and 11.3, with the largest drop on 16:9-squeezed panoramas, where the embedding carries the content aspect. Stage 2, where the embedding enters, was trained in two runs that differ only in it (Table 10, Appendix D): the run with the embedding leads by 3.2 mAA@30, 3.7 AUC@30 and 9.9 AUC@30 on 16:9 panoramas, trails by 1.7 on the laser panorama track, and is the one carried into Stage 3.

Table 4: Ablations. Top: the final weights with one input change at a time; the crop-loader row routes panoramas by the dataset annotation, with which the full model scores as with the detector (Table 8). Bottom: the two Stage-2 runs, identical except for the aspect-ratio embedding (Table 10). Columns: per-tuple mAA@30 on the 88 heterogeneous 2D3DS tuples; laser mixed-track AUC@30; AUC@30 on 2D3DS panorama tuples squeezed to 16:9.
<table><tr><td></td><td>2D3DS mAA@30</td><td>Laser mixed AUC@30</td><td>16:9 panoramas AUC@30</td></tr><tr><td>Final weights (full model)</td><td>80.4</td><td>79.4</td><td>78.6</td></tr><tr><td>Crop loader instead of full-field-of-view resizing</td><td>77.1</td><td>74.4</td><td></td></tr><tr><td>Aspect-ratio embedding withheld</td><td>57.2</td><td>68.1</td><td>42.4</td></tr><tr><td>Stage-2 run with the embedding (carried into Stage 3)</td><td>63.8</td><td>65.4</td><td>63.8</td></tr><tr><td>Stage-2 run without the embedding</td><td>60.6</td><td>61.7</td><td>53.9</td></tr></table>

## 5 CONCLUSION

We presented MEOW, a feed-forward model that reconstructs metric pointmaps and camera poses from one tuple of perspective, fisheye and full-panorama images in a single forward pass, with no camera information supplied for any view. A perspective-pretrained backbone is adapted, with full-field-of-view resizing, on LENSCOPE tuples rendered through a continuous manifold of camera models with exact rays and depth and checked for covisibility. Adapted using synthetic tuples only, the model transfers zero-shot: 80.4 mAA@30 on heterogeneous 2D3DS tuples against 54.3 for Wid3R given the camera type of every view, and every tuple on the four-view mixed track of our laser-scanned benchmark registered at 79.4 AUC@30. The ablations show what it relies on: the fullfield-of-view resizing and the aspect-ratio embedding at inference, and the embedding already at the stage where it enters training. The shared metric world also yields cross-camera correspondences without a matcher (Appendix E).

Limitations. The engine covers synthetic indoor scenes, and the final model is adapted on roomscale tuples of up to eight views: on 2D3DS tuples of 15–24 views it trails Wid3R, which puts Wid3R 0.5 mAA@30 ahead when pairs are pooled over all tuples (Appendix F), and on centimetre-baseline perspective video (Replica (Straub et al., 2019), ADT (Pan et al., 2023)) it trails the perspective models (Table 12, Appendix D). The scale-invariant metrics do not test the predicted metric scale, which is 12–25% short of the truth (Appendix F). Extending the training range to longer sequences and replaying real perspective video during adaptation are the next steps.

## AI USE STATEMENT

The research ideas, the experimental design, the claims and the story of this paper are the authors’. Generative AI tools assisted with the manuscript and citation checks, LAT X editing, schematic artwork, and benchmarking code. The authors verified the code, re-derived every reported number from the raw logs, checked the citations, reviewed the final manuscript, and take responsibility for the final content. No generative model was used to produce training data.

## REPRODUCIBILITY STATEMENT

All randomness in the engine, the training runs and the benchmark construction is seeded (the sampler’s feasibility tables also depend on Python’s unrecorded string-hash seed; the tables used in training will be released), and both scene generations regenerate from seeds with scripts that will be released with the code: a regeneration of three first-generation scenes reproduced the rays, depth and validity masks of the original training packs exactly, with RGB differing only by renderer sampling noise (PSNR above 75 dB). The engine, the benchmark construction scripts with their convention tests, the evaluation protocols and the training configurations are planned for public release. Model checkpoints are planned for public release upon acceptance.

## REFERENCES

Iro Armeni, Sasha Sax, Amir R. Zamir, and Silvio Savarese. Joint 2D-3D-semantic data for indoor scene understanding. arXiv preprint arXiv:1702.01105, 2017.

Jonathan T. Barron. A general and adaptive robust loss function. In CVPR, 2019.

Paul J. Besl and Neil D. McKay. A method for registration of 3-D shapes. IEEE Transactions on Pattern Analysis and Machine Intelligence, 14(2):239–256, 1992.

Oleksandr Bogdan, Viktor Eckstein, Franc¸ois Rameau, and Jean-Charles Bazin. DeepCalib: A deep learning approach for automatic intrinsic calibration of wide field-of-view cameras. In CVMP, 2018.

Vukasin Bozic, Isidora Slavkovic, Dominik Narnhofer, Nando Metzger, Denis Rozumny, Konrad Schindler, and Nikolai Kalischek. Unified panoramic geometry estimation via multi-view foundation models. arXiv preprint arXiv:2605.26368, 2026.

Yohann Cabon, Lucas Stoffl, Leonid Antsfeld, Gabriela Csurka, Boris Chidlovskii, Jer´ ome Revaud,ˆ and Vincent Leroy. MUSt3R: Multi-view network for stereo 3D reconstruction. In CVPR, 2025. arXiv:2503.01661.

Angel Chang, Angela Dai, Thomas Funkhouser, Maciej Halber, Matthias Nießner, Manolis Savva, Shuran Song, Andy Zeng, and Yinda Zhang. Matterport3D: Learning from RGB-D data in indoor environments. In 3DV, 2017.

Boris Chidlovskii, Victor Domsa, Gabriela Csurka, and Levente Tamas. UCE: Universal camera embeddings for scalable 3D scene understanding. In ECCV Workshops, 2026.

T. A. Clarke, X. Wang, and J. G. Fryer. The principal point and CCD cameras. The Photogrammetric Record, 16(92):293–312, 1998. doi: 10.1111/0031-868X.00127.

Matt Deitke, Eli VanderBilt, Alvaro Herrasti, Luca Weihs, Jordi Salvador, Kiana Ehsani, Winson Han, Eric Kolve, Ali Farhadi, Aniruddha Kembhavi, and Roozbeh Mottaghi. ProcTHOR: Largescale embodied AI using procedural generation. In NeurIPS, 2022.

Ruxiao Duan, Erin Hong, Dongxu Zhao, Eric Turner, Alex Wong, and Yunwen Zhou. Fisheye3R: Adapting unified 3D feed-forward foundation models to fisheye lenses. In European Conference on Computer Vision (ECCV), 2026. arXiv:2603.28896.

Girish Chandar Ganesan, Yuliang Guo, Liu Ren, and Xiaoming Liu. UniDAC: Universal metric depth estimation for any camera. In CVPR, 2026. arXiv:2603.27105.

Suchisrit Gangopadhyay, Jung Hee Kim, Xien Chen, Patrick Rim, Hyoungseob Park, and Alex Wong. Extending foundational monocular depth estimators to fisheye cameras with calibration tokens. In ICCV, 2025. arXiv:2508.04928.

Robert Geirhos, Jorn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel,¨ Matthias Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2:665–673, 2020.

Christopher Geyer and Kostas Daniilidis. A unifying theory for central panoramic systems and practical implications. In ECCV, 2000.

Ryan Griffiths and Donald G. Dansereau. RoRE: Rotary ray embedding for generalised multi-modal scene understanding. In ICLR, 2026.

Yijing Guo, Mengjun Chao, Luo Wang, Tianyang Zhao, Haizhao Dai, Yingliang Zhang, Jingyi Yu, and Yujiao Shi. PanoVGGT: Feed-forward 3D reconstruction from panoramic imagery. In CVPR, 2026. arXiv:2603.17571.

Yuliang Guo, Sparsh Garg, S. Mahdi H. Miangoleh, Xinyu Huang, and Liu Ren. Depth any camera: Zero-shot metric depth estimation from any camera. In CVPR, 2025. arXiv:2501.02464.

Namitha Guruprasad, Abhay Yadav, Cheng Peng, and Rama Chellappa. CAM3R: Camera-agnostic model for 3D reconstruction. In European Conference on Computer Vision (ECCV), 2026. arXiv:2603.22631.

Yannick Hold-Geoffroy, Dominique Piche-Meunier, Kalyan Sunkavalli, Jean-Charles Bazin,´ Franc¸ois Rameau, and Jean-Franc¸ois Lalonde. A perceptual measure for deep single image camera and lens calibration. IEEE Transactions on Pattern Analysis and Machine Intelligence, 45(9): 10603–10614, 2023. doi: 10.1109/TPAMI.2023.3269641.

Tingjun Huang, Dmitry Rudshin, Mathieu Meyer, Pietro Bonazzi, Marc Pollefeys, and Emilia Szymanska. RIGOR: Rig-informed geometry for omnidirectional reconstruction.´ arXiv preprint arXiv:2609.13504, 2026.

Wonbong Jang, Philippe Weinzaepfel, Vincent Leroy, Lourdes Agapito, and Jer´ ome Revaud.ˆ Pow3R: Empowering unconstrained 3D reconstruction with camera and scene priors. In CVPR, 2025. arXiv:2503.17316.

Yuzhou Ji, Xiaotian Yang, and Zhipeng Zhang. CasaMaestro: Multi-view panoramas for house-scale 3D reconstruction. In Computer Vision – ECCV 2026, 2026. doi: 10.1007/978-3-032-37252-9 12.

Linyi Jin, Jianming Zhang, Yannick Hold-Geoffroy, Oliver Wang, Kevin Blackburn-Matzen, Matthew Sticha, and David F. Fouhey. Perspective fields for single image camera calibration. In CVPR, 2023.

Dongki Jung, Jaehoon Choi, Adil Qureshi, Somi Jeong, Dinesh Manocha, and Suyong Yeon. Wid3R: Wide field-of-view 3D reconstruction via camera model conditioning. In European Conference on Computer Vision (ECCV), 2026. arXiv:2602.05321.

Juho Kannala and Sami S. Brandt. A generic camera model and calibration method for conventional, wide-angle, and fish-eye lenses. IEEE TPAMI, 28(8):1335–1340, 2006.

Nikhil Keetha, Norman Muller, Johannes Sch¨ onberger, Lorenzo Porzi, Yuchen Zhang, Tobias Fis-¨ cher, Arno Knapitsch, Duncan Zauss, Ethan Weber, Nelson Antunes, Jonathon Luiten, Manuel Lopez-Antequera, Samuel Rota Bulo, Christian Richardt, Deva Ramanan, Sebastian Scherer, and\` Peter Kontschieder. MapAnything: Universal feed-forward metric 3D reconstruction. In 3DV, 2026. arXiv:2509.13414.

Bogdan Khomutenko, Gaetan Garcia, and Philippe Martinet. An enhanced unified camera model.¨ IEEE Robotics and Automation Letters, 1(1):137–144, 2016.

Vincent Leroy, Yohann Cabon, and Jer´ ome Revaud. Grounding image matching in 3D withˆ MASt3R. In ECCV, 2024.

Boying Li, Cheng Zhang, Weirong Chen, Guyuan Chen, Daniel Cremers, Jianfei Cai, Ian Reid, and Hamid Rezatofighi. CalibAnyView: Beyond single-view camera calibration in the wild. arXiv preprint arXiv:2605.14615, 2026a. URL https://arxiv.org/abs/2605.14615v2.

Haodong Li, Wangguangdong Zheng, Jing He, Yuhao Liu, Xin Lin, Xin Yang, Ying-Cong Chen, and Chunchao Guo. DA<sup>2</sup>: Depth anything in any direction. In International Conference on Learning Representations (ICLR), 2026b. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ 786e39208b37bfcb0ad89413f155df99-Abstract-Conference.html.

Samuel Li, Pujith Kachana, Prajwal Chidananda, Saurabh Nair, Yasutaka Furukawa, and Matthew Brown. Rig3R: Rig-aware conditioning for learned 3D reconstruction. In NeurIPS, 2025. arXiv:2506.02265.

Xi Li, Linyuan Li, Yan Wu, Tong Rao, Kai Zhang, Xinchen Hui, and Cihui Pan. Argus: Metric panoramic 3D reconstruction for indoor scenes. In Computer Vision – ECCV 2026, 2026c. doi: 10.1007/978-3-032-37369-4 34.

Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Yang Zhao, Sida Peng, Hengkai Guo, Xiaowei Zhou, Guang Shi, Jiashi Feng, and Bingyi Kang. Depth anything 3: Recovering the visual space from any views. In International Conference on Learning Representations (ICLR), 2026a. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ e4cd50120b6d7e8daff1749d6bbaa889-Abstract-Conference.html.

Xin Lin, Meixi Song, Dizhe Zhang, Wenxuan Lu, Haodong Li, Bo Du, Ming-Hsuan Yang, Truong Nguyen, and Lu Qi. Depth any panoramas: A foundation model for panoramic depth estimation. In CVPR, 2026b. arXiv:2512.16913.

Jiale Liu, Jungang Li, Jieming Yu, Xinglin Yu, Zihao Dongfang, Zongjian Ding, Kaifeng Ding, Yi Yang, Lidong Chen, Yang Zou, Shunwen Bai, Jiahuan Zhang, Haoran Huang, Shan Huang, Yudong Gao, and Mingjun Cheng. CM-EVS: Sparse panoramic RGB-D-pose data for complete scene coverage. arXiv preprint arXiv:2605.15597, 2026.

Yaroslava Lochman, Kostiantyn Liepieshov, Jianhui Chen, Michal Perdoch, Christopher Zach, and James Pritts. BabelCalib: A universal approach to calibrating central cameras. In ICCV, 2021.

Zeyu Ma, Alexander Raistrick, and Jia Deng. SimpleProc: Fully procedural synthetic data from simple rules for multi-view stereo. arXiv preprint arXiv:2604.04925, 2026.

Kanti V. Mardia and Peter E. Jupp. Directional Statistics. Wiley, 2000.

Christopher Mei and Patrick Rives. Single view point omnidirectional camera calibration from planar grids. In ICRA, 2007.

Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, et al. DINOv2: Learning robust visual features´ without supervision. TMLR, 2024.

Jing Ou, Zidong Cao, Yinrui Ren, Zhuoxiao Li, Jinjing Zhu, Tongyan Hua, Shuai Zhang, Hui Xiong, and Wufan Zhao. Holo360D: A large-scale real-world dataset with continuous trajectories for advancing panoramic 3D reconstruction and beyond. In European Conference on Computer Vision (ECCV), 2026. URL https://arxiv.org/abs/2604.22482.

Xiaqing Pan, Nicholas Charron, Yongqian Yang, Scott Peters, Thomas Whelan, Chen Kong, Omkar Parkhi, Richard Newcombe, and Yuheng Carl Ren. Aria Digital Twin: A new benchmark dataset for egocentric 3D machine perception. In ICCV, 2023.

Photis Patonis. Methodology and tool development for mobile device cameras calibration and evaluation of the results. Sensors, 23, 2023.

Luigi Piccinelli, Christos Sakaridis, Mattia Segu, Yung-Hsu Yang, Siyuan Li, Wim Abbeloos, and Luc Van Gool. UniK3D: Universal camera monocular 3D estimation. In CVPR, 2025.

Rene Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction.´ In IEEE/CVF International Conference on Computer Vision (ICCV), 2021.

Enoc Sanz-Ablanedo, Jose Ram´ on Rodr´ ´ıguez-Perez, Julia Armesto, and Mar´ ´ıa Flor Alvarez<sup>´</sup> Taboada. Geometric stability and lens decentering in compact digital cameras. Sensors, 10, 2010.

Daniil Sinitsyn, Nikita Araslanov, and Daniel Cremers. RayTun3R: Online camera adaptation in 3D foundation models. arXiv preprint arXiv:2607.02711, 2026.

Meixi Song, Dizhe Zhang, Hao Ren, Ruiyang Zhang, Bo Du, Ming-Hsuan Yang, and Lu Qi. UniSHARP: Universal sharp monocular view synthesis. arXiv preprint arXiv:2606.07514, 2026.

Julian Straub et al. The Replica dataset: A digital replica of indoor spaces. arXiv preprint arXiv:1906.05797, 2019.

Javier Tirado-Gar´ın and Javier Civera. AnyCalib: On-manifold learning for model-agnostic singleview camera calibration. In ICCV, 2025. arXiv:2503.12701.

Shinji Umeyama. Least-squares estimation of transformation parameters between two point patterns. IEEE TPAMI, 13(4):376–380, 1991.

Vladyslav Usenko, Nikolaus Demmel, and Daniel Cremers. The double sphere camera model. In 3DV, 2018.

Alexander Veicht, Paul-Edouard Sarlin, Philipp Lindenberger, and Marc Pollefeys. GeoCalib: Learning single-image calibration with geometric optimization. In ECCV, 2024.

Jianyuan Wang, Christian Rupprecht, and David Novotny. PoseDiffusion: Solving pose estimation via diffusion-aided bundle adjustment. In ICCV, 2023.

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, and David Novotny. VGGT: Visual geometry grounded transformer. In CVPR, 2025a.

Qianqian Wang, Yifei Zhang, Aleksander Holynski, Alexei A. Efros, and Angjoo Kanazawa. Continuous 3D perception model with persistent state. In CVPR, 2025b. arXiv:2501.12387.

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jer´ ome Revaud. DUSt3R:ˆ Geometric 3D vision made easy. In CVPR, 2024.

Xintao Wang, Liangbin Xie, Chao Dong, and Ying Shan. Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data. In ICCV Workshops, 2021.

Yifan Wang, Jianjun Zhou, Haoyi Zhu, Wenzheng Chang, Yang Zhou, Zizun Li, Junyi Chen, Jiangmiao Pang, Chunhua Shen, and Tong He. π<sup>3</sup>: Permutation-equivariant visual geometry learning. In ICLR, 2026. arXiv:2507.13347.

Zhuang Xiong, Guohao Zhang, Chen Zhang, Zheyu Jiang, Yuchao Mei, Qingshan Xu, and Wenbing Tao. Look up and look back: Hidden attention and latent orientation in a frozen foundation mode for panoramic SLAM. arXiv preprint arXiv:2608.00925, 2026.

Jianing Yang, Alexander Sax, Kevin J. Liang, Mikael Henaff, Hao Tang, Ang Cao, Joyce Chai, Franziska Meier, and Matt Feiszli. Fast3R: Towards 3D reconstruction of 1000+ images in one forward pass. In CVPR, 2025. arXiv:2501.13928.

Botao Ye, Marc Pollefeys, Ming-Hsuan Yang, and Abhijit Kundu. OmniPoint: Universal monocular metric pointcloud from any camera. In Computer Vision – ECCV 2026, 2026. doi: 10.1007/ 978-3-032-37602-2 32.

Shuo Zhang, Xin Su, Wei Wang, Jun Liu, Xinrui Zeng, Yongsen Chen, Chenjie Wang, Guibo Zhu, Jinqiao Wang, Bin Luo, and Liangpei Zhang. G-ray: Ray-level relative geometric position encoding in multi-view vision transformers under camera heterogeneity. arXiv preprint arXiv:2609.15018, 2026.

Zhiwei Zhang, Ruikai Xu, Weijian Zhang, Zhizhong Zhang, Xin Tan, Jingyu Gong, Yuan Xie, and Lizhuang Ma. PFDepth: Heterogeneous pinhole-fisheye joint depth estimation via distortionaware gaussian-splatted volumetric fusion. In ACM Multimedia, 2025. arXiv:2509.26008.

Jia Zheng, Junfei Zhang, Jing Li, Rui Tang, Shenghua Gao, and Zihan Zhou. Structured3D: A large photo-realistic dataset for structured 3D modeling. In ECCV, 2020.

Heng Zhou, Shuhong Liu, Yonghao He, Bohao Zhang, Fa Fu, Chenhui Hou, Xianbao Hou, Lijun Han, Cong Yang, and Wei Sui. X-Lens: Real-time metric depth estimation with heterogeneous cameras. arXiv preprint arXiv:2607.12993, 2026.

## A TRAINING RUNS AND EVALUATION CONVENTIONS

The final model descends from the public MapAnything checkpoint through domain adaptation on first-generation renders (35 epochs), Stage 2 with online camera sampling (100 epochs), and two Stage-3 runs on second-generation scenes: a 15-epoch run on an earlier 189-scene batch of the same generator rendered at $2 0 4 8 \times 1 0 2 4$ , and a run on the 469-scene training split, trained in segments of at most 125 epochs, of which the checkpoint after 859 epochs is used; the final weights interpolate the two with weights 0.25 and 0.75; both parents and the interpolation script will be released upon acceptance. Tables 9 and 11 also report the checkpoint after the first 90 epochs of the long run and an earlier interpolation with the same weights whose late parent is the checkpoint after 359 epochs. Stage 2 gives each view of a tuple its own bucket, in a fixed pattern set by the batch bucket and K; Stage 3 and all evaluations use one bucket per tuple. All stages use AdamW with weight decay 0.05 and cosine decay to 1% of the peak learning rate. Stage 1 follows the official schedule, with peak rates of $1 0 ^ { - 4 } ~ ( \dot { 5 } \times 1 0 ^ { - 6 }$ for the encoder) and 7 warm-up epochs; Stages 2 and 3 use $1 . 6 \times \dot { 1 } 0 ^ { - 5 }$ $( 8 \times 1 0 ^ { - 7 }$ for the encoder) in bf16 autocast, with 15 warm-up epochs in Stage 2 and one in Stage 3. Each segment of the long run is a new run started from the weights of the previous segment, so the schedule restarts in every segment. Reported results use isolated checkpoint evaluation and explicit preprocessing and panorama-routing settings; scripts and logs will be included in the release.

## B LENSCOPE IN DETAIL

This appendix gives the parameters behind Section 3.2 and the complete procedure for one training tuple (Algorithm 1).

Renders. First-generation panoramas are rendered at 2160 × 1080; the 37,010 second-generation poses at 3072 × 1536 with 128 samples per pixel. In the first generation, narrow rectilinear views are re-rendered from the sharpest pinhole pack (18–40 pixels per degree) that covers the target and from the panorama (6 pixels per degree) otherwise; the second generation renders every view from its panorama (8.5 pixels per degree; 5.7 for the 2048 × 1024 batch of the 15-epoch run).

Camera models. Table 5 lists the seven models with their sampling weights and field-of-view ranges. The weights give a rectilinear majority with a deliberate share of fisheye and panoramic views.

Table 5: Camera models of the manifold with their sampling weights and initial field-of-view ranges. Rectilinear draws are log-normal around 80<sup>◦</sup> (log-space standard deviation 0.32) and clipped to the listed range; fisheye models and spherical crops are drawn log-uniformly. The parameter is diagonal for rectilinear models, nominal across the image circle for fisheye models (equidistant focal length, before distortion) and horizontal for spherical crops. Tuples in the aimed mode raise the lower bound to 90<sup>◦</sup>, and the source-resolution floor and the covisibility repair can adjust a draw. Full panoramas span 360<sup>◦</sup> and receive a random SO(3) content rotation.
<table><tr><td>Camera model</td><td>Family</td><td>Sampling weight</td><td>Field-of-view parameter</td></tr><tr><td>OpenCV</td><td>Rectilinear with distortion</td><td>0.40</td><td>48-120°</td></tr><tr><td>Pinhole</td><td>Rectilinear</td><td>0.15</td><td>48-95°</td></tr><tr><td>Fisheye624</td><td>Fisheye</td><td>0.13</td><td>120-210°</td></tr><tr><td>EUCM</td><td>Fisheye</td><td>0.07</td><td>90-180°</td></tr><tr><td>Mei</td><td>Fisheye</td><td>0.05</td><td>140-200°</td></tr><tr><td>Spherical crop</td><td>Spherical</td><td>0.05</td><td>110-300°</td></tr><tr><td>Full panorama</td><td>Spherical</td><td>0.15</td><td>360°</td></tr></table>

Photographic priors. Principal-point shifts: 90% Gaussian with $\sigma = 1 . 2 \%$ of the image size clipped at ±5%, of the order of the decentering reported for compact and phone cameras (Sanz-Ablanedo et al., 2010; Patonis, 2023), and 10% uniform to ±22% to cover zoom lenses and offcentre crops (Clarke et al., 1998; Jin et al., 2023). Roll: 90% a zero-centred two-component Cauchy mixture (scales 0.06<sup>◦</sup> and 5.7<sup>◦</sup>, weights 1/3 and 2/3) inspired by the roll model of Hold-Geoffroy et al. (2023), clipped at ±30<sup>◦</sup>, 10% uniform over 0–360<sup>◦</sup>; tilt: 90% Rayleigh with $\sigma = 4 ^ { \circ }$ clipped at $1 2 ^ { \circ }$ , our choice for a near-level indoor prior, 10% uniform over $0 { - } 1 2 ^ { \circ }$ . Optics layer: modulationtransfer randomisation on half of the views (downsampling factor log-uniform in [0.35, 1]), lateral chromatic aberration, $\cos ^ { 4 }$ vignetting and sensor noise, a synthetic degradation pipeline in the manner of Wang et al. (2021); without it, sharpness would reveal the source asset and hence the field of view, a shortcut in the sense of Geirhos et al. (2020). The mutual information between field of view and sharpness, estimated on the first-generation source mix, falls from 0.557 to 0.251 bits.

Covisibility check and repair. The check renders the sampled views at 96 pixels on the long side with 2× supersampling and recomputes $c _ { i j }$ with Eq. (2). A tuple whose graph is not connected at 0.25 is repaired by moving its disconnected views in three steps (0.5, 0.9 and 1.0 of the way in total) toward a wide field of view, a square aspect and no tilt or principal-point offset. A tuple that still fails is rejected and the scene redrawn once; after that the sample falls back to the unaugmented renders, selected by connected walks on their precomputed covisibility (0.25), or by per-camerapair thresholds when the walks fail (10.6% of logged Stage-2 tuples, 14% of views). Kept views are rendered once more at their target resolution with 2× supersampling.

Tuple-size schedule. The first epoch uses $K = 4$ only, and a 15-epoch cosine ramp opens the distribution to its target over $K = \overbracket { 2 } , \ldots , 8 ( 0 . 0 8 , 0 . 1 2 , 0 . \overbar { 1 } 5 , 0 . 1 8 , 0 . 1 8 , 0 . 1 5 , 0 . 1 4 )$ . Each step holds $\lfloor 4 8 / K \rfloor$ tuples per GPU on four GPUs (at most 48 images) and $\lceil \lfloor 4 8 / K \rfloor / 2 \rceil$ on eight (at most 25 images).

Algorithm 1 One training tuple: from panorama renders to the network input   
Require: scene with panorama renders $\{ E _ { k } \}$ (RGB, rays, depth, mask) and offline covisibility $c _ { k l } ;$ tuple size   
K; bucket $( h , w ) \bar { \in { \cal B } }$   
Ensure: input tensors $\{ \tilde { I } _ { i } , a _ { i } , \pi _ { i } \} _ { i = } ^ { K }$ and targets $\{ \mathbf { r } _ { i } , d _ { i } , \mathbf { R } _ { i } , \mathbf { t } _ { i } , m _ { i } \} _ { i = 1 } ^ { K }$   
1: draw the plan mode: aimed with probability 0.55 (converging on a common surface point with probability   
0.90, parallel toward it otherwise; fields of view from $9 0 ^ { \circ } )$ , native orientation otherwise   
2: $( k _ { 1 } , \dot { \ldots } , k _ { K } ) \gets \mathrm { C o N N E C T E D W a L K } ( c , K , \tau = 0 . 4 0 ) ;$ ; in the aimed mode relax to $\tau { = } 0 . 2 0$ if no chain of   
length K exists; if no plan exists, retry in the other mode ▷ distinct optical centres   
3: for $i = 1$ to $K$ do   
4: draw the camera model $\mu _ { i }$ , field of view $\theta _ { i } ,$ , distortion, principal point, roll and tilt from Table 5 and the   
priors above   
5: ${ \bf R } _ { i } ^ { \mathrm { c } } \gets \mathrm { g a z e }$ rotation or native orientation; $\mathbf { i f } \ \mu _ { i }$ is a full panorama then ${ \bf R } _ { i } ^ { \mathrm { c } } \sim S O ( 3 )$   
6: $\boldsymbol { V _ { i } } \gets \bar { \mathrm { R E S A M P L E } } ( E _ { k _ { i } } , \mu _ { i } , \theta _ { i } , \mathbf { R } _ { i } ^ { \mathrm { c } } , 9 6 \ : \mathrm { p x } )$ ▷ rays analytically; RGB, depth and mask by ray lookup   
7: end for   
8: $c _ { i j } \gets \mathrm { C o v I S } ( \{ V _ { i } \} )$ by Eq. $( 2 ) ; { \mathcal { G } }$ ← graph with edges $c _ { i j } > 0 . 2 5$   
9: for $f \in ( 0 . 5 , 0 . 8 , 1 . 0 )$ while G is not connected do   
10: pull each view outside the largest component a fraction f of the remaining way toward the repair target   
(wide field of view, square aspect, no tilt, centred); re-render at 96 px; recompute $\mathbf { \bar { \boldsymbol { g } } }$   
11: end for   
12: if G is not connected then return REJECT ▷ redraw the scene once, then fall back to the native renders   
13: end if   
14: for $i = 1$ to K do   
15: V ← RESAMPLE $( E _ { k _ { i } } , \mu _ { i } , \theta _ { i } , { \bf R } _ { i } ^ { \mathrm { c } } .$ target resolution, 2× supersampling)   
16: $( H _ { i } , W _ { i } ) $ shape of $V _ { i } ; a _ { i }  \log ( W _ { i } / H _ { i } ) ; \pi _ { i } $ [azimuth span of the middle row $> 3 5 0 ^ { \circ } ] ; \mathbf { i f } \pi _ { i }$   
then $a _ { i } \gets \log 2$   
17: $\tilde { I } _ { i } \gets \mathsf { R G B }$ of $V _ { i }$ resized to $( h , w )$ ; rays, depth and mask resampled through the same map (nearest for   
depth and mask)   
18: end for   
19: return $\{ \tilde { I } _ { i } , a _ { i } , \pi _ { i } \} _ { i = 1 } ^ { K }$ and the targets

At inference only the last loop runs, on the real images, with the bucket chosen from the mean aspect ratio of the tuple and $\pi _ { i }$ from the detector of Appendix C.

## C PANORAMA DETECTOR

The image is resized to 128 rows (width in proportion). With $\mathrm { M A E } ( \cdot , \cdot )$ the mean absolute colour difference between two columns and σ¯ the mean horizontal colour standard deviation of a band of

rows,

$$
\rho _ { \mathrm { s e a m } } = \frac { \mathrm { M A E } \left( I _ { : , W } , I _ { : , 1 } \right) } { P _ { 9 0 } \{ \mathrm { M A E } \left( I _ { : , k } , I _ { : , k + 1 } \right) \} _ { k < W } } , \qquad \rho _ { \mathrm { p o l e } } = \frac { \frac { 1 } { 2 } \left( \bar { \sigma } _ { \mathrm { t o p } } + \bar { \sigma } _ { \mathrm { b o t t o m } } \right) } { \bar { \sigma } _ { \mathrm { m i d d l e } } } ,\tag{6}
$$

over the top and bottom $H / 1 6$ rows and the middle $H / 8$ rows. In a full panorama the last and first columns are adjacent in content, so $\rho _ { \mathrm { s e a m } }$ is small, and the pole rows compress all longitudes into near-constant colour, so $\rho _ { \mathrm { p o l e } }$ is small. The guard η is the fraction of rows in the central half whose 1%-wide left and right edge strips both hold mostly non-black pixels; it rejects circular fisheyes, whose black corners would satisfy both ratios. The flag is $\pi _ { i } = [ \bar { \eta } \geq 0 . 5 ] \wedge [ \rho _ { \mathrm { s e a m } } < 3 ] \wedge [ \rho _ { \mathrm { p o l e } } <$ 0.55]. Table 6 lists the error counts on the two headline sets and, in its caption, how the other sets are routed.

Table 6: Full-panorama detector audit on the two headline sets with our detector (two image cues plus the black-circle guard, thresholds fixed on training renders): panoramas missed and other views flagged as panoramas. The 2D3DS panorama tuples are also routed with the detector; single panoramas and Matterport3D are flagged as panoramas throughout and the perspective video sets not at all. Counts on the other laser tracks, the 2D3DS panorama tuples and the single panoramas will accompany the code release.
<table><tr><td>Set</td><td></td><td>not detected)</td><td>Views Misses (panoramas False alarms (other views flagged)</td></tr><tr><td>Heterogeneous 2D3DS tuples, with</td><td>503</td><td>0 /189</td><td>2/314</td></tr><tr><td>guard Laser mixed track, with guard</td><td>96</td><td>0 /24</td><td>0/72</td></tr></table>

## D ADDITIONAL TABLES AND FIGURES

Table 7 contrasts the training data of the compared methods with LENSCOPE, Table 8 lists the resizing and routing controls of the final model, Table 9 reports real panoramas, Table 10 the Stage-2 runs with and without the aspect-ratio embedding, Table 11 squeezed inputs of real panoramas, Table 12 the real perspective video sets, Table 13 the timing measurements, and Table 14 the 2D3DS results by tuple size. Figure 5 shows the camera models of LENSCOPE on one scene, Figure 6 the as-fed distributions of the training stream. Figure 7 shows the resizing on one tuple, and Figure 8 the laser-scanned benchmark’s twelve stations with one of its tuples.

![](images/f81fa321c33153de16c627c6e251462875ed54e0f37177e6f30979fc7eed82c0.jpg)  
Figure 5: Same-scene camera-model gallery from LENSCOPE: two poses of one procedural scene (rows) rendered through five camera models (columns) at fixed illustrative fields of view; panel titles use the descriptive model names (Brown–Conrady for OpenCV, extended unified for EUCM). The gallery illustrates the camera models; training tuples are assembled separately under the covisibility check of Section 3.2.

Table 7: Training-data construction of the compared methods, from released code where available (X-Lens, a calibrated-rig depth method, omitted; test-time camera inputs are in Table 1). Of the compared methods, LENSCOPE draws each view’s camera from a continuous manifold, checks covisibility after the cameras are drawn, and uses no real image after the public perspective pretraining.
<table><tr><td></td><td>Wid3R</td><td>CAM3R</td><td>Fisheye3R</td><td></td><td>PanoVGGT</td><td></td><td>LENSCOPE (ours)</td></tr><tr><td>Training data</td><td></td><td>4 real + 5 synthetic sets 4 real sets + views de- 3 real + 5 synthetic sets 2 real + 2 syn- Synthetic only, two proce- rived from panoramas</td><td></td><td></td><td></td><td>thetic panorama dural generations</td><td></td></tr><tr><td>Camera models Pinhole, seen</td><td>Mei, EUCM, Fisheye624, OpenCV splatted from</td><td>Fisheye624, Pinhole, fisheye (Aria; Pinhole; equirectangu- equidistant lar panorama (ERP); panoramas), ERP</td><td></td><td>from Brandt fisheye distorted online from perspective frames; no 360°</td><td>Kannala- ERP only</td><td>Continuous</td><td>manifold: pinhole, OpenCV, Fish- eye624, EUCM, Mei, spherical crops, full ERP</td></tr><tr><td>tity within a tuple</td><td>pinhole (25%) Camera iden- One class per sample</td><td>Two-view pairs of Per-frame mixed classes</td><td>p=0.5</td><td>distortion, N/A</td><td></td><td>views, centres</td><td>Drawn per view, 2–8 distinct optical</td></tr><tr><td>Covisibility control</td><td>lap test</td><td>Position prior, no over- Offline baseline and an- GT-depth gle thresholds</td><td>tion</td><td>≥ 25% before distor- tory grouping</td><td>overlap Room or trajec- Checked after the cam-</td><td>projection, depth agree- ment); failures repaired or</td><td>eras are drawn (mutual</td></tr></table>

Table 8: Resizing and routing controls for the final checkpoint and the public MapAnything weights on the two mixed-camera benchmarks; 2D3DS pose metrics per tuple, averaged over tuples. The Stage-1 row is the checkpoint adapted on first-generation renders with the public loader and loss, before the full-field-of-view (full-FoV) resizing, the embedding and the camera manifold. The detector reproduces the dataset annotation on every tuple of the mixed laser track (189 of 189 panoramas found on 2D3DS, 2 of 314 other views flagged). On the single-camera laser tracks the annotation gives 71.0 / 86.6 / 73.2 AUC@30 (panorama / pinhole / fisheye) against the detector’s 71.0 / 85.7 / 73.2; on the 16 panorama tuples inside the covisibility envelope of Appendix F the final model reaches 92.6.
<table><tr><td></td><td></td><td colspan="2">2D3DS tuples</td><td colspan="3">Laser mixed track</td></tr><tr><td>Method</td><td>Input resizing and panorama flag</td><td>mAA@30</td><td>ATE ↓</td><td>AUC@30</td><td>Acc ↓</td><td>Comp ↓</td></tr><tr><td>MEOW</td><td>Full-FoV resizing, detector flag</td><td>80.4</td><td>0.62</td><td>79.4</td><td>0.142</td><td>0.367</td></tr><tr><td>MEOW</td><td>Full-FoV resizing, annotation flag</td><td>80.4</td><td>0.62</td><td>79.4</td><td>0.142</td><td>0.367</td></tr><tr><td>MEOW</td><td>Public crop loader, annotation flag</td><td>77.1</td><td>0.75</td><td>74.4</td><td>0.168</td><td>0.349</td></tr><tr><td>Stage-1 checkpoint (renders only)</td><td>Public crop loader</td><td>68.3</td><td>0.99</td><td>69.0</td><td>0.120</td><td>0.322</td></tr><tr><td>MapAnything</td><td>Public crop loader</td><td>16.4</td><td>1.48</td><td>24.5</td><td>0.255</td><td>0.830</td></tr><tr><td>MapAnything</td><td>Full-FoV resizing, diagnostic</td><td>15.4</td><td>1.45</td><td>19.7</td><td>0.272</td><td>0.808</td></tr></table>

Table 9: Real panoramas. 2D3DS tuples: same-room tuples of 2–13 panoramas, 19 cases, seven areas; Wid3R reaches RRA@30 94.0 and RTA@30 96.1, PanoVGGT 100 and 100. Single panorama: 40 panoramas of areas 5a and 5b, per-view Sim(3)-aligned Chamfer L1 in metres and median relative depth error. Matterport3D (MP3D) tuples: 18 scans, 8 panoramas per tuple, pointmap accuracy and completeness in metres. <sup>†</sup>PanoVGGT trains on 2D3DS and Matterport3D, so its 2D3DS numbers are in-domain; our model, Wid3R and the open baselines are zero-shot here. Wid3R’s published 79.9 on 2D3DS uses a different protocol and is not the number in this column.
<table><tr><td colspan="2"></td><td>2D3DS tuples</td><td colspan="2">2D3DS single panorama</td><td>MP3D tuples</td></tr><tr><td>Model</td><td>Checkpoint</td><td>AUC@30</td><td>Chamfer L1 ↓</td><td>Depth rel. ↓</td><td>Acc / Comp ↓</td></tr><tr><td>MEOW</td><td>Final</td><td>75.4</td><td>0.296</td><td>0.195</td><td>0.237 / 0.962</td></tr><tr><td>MEOW</td><td>90-epoch run</td><td>70.5</td><td>0.328</td><td>0.315</td><td>0.249 / 0.952</td></tr><tr><td>Wid3R (camera type given)</td><td>Released</td><td>92.2</td><td>0.090</td><td>0.044</td><td></td></tr><tr><td>(Jung et al., 2026) PanoVGGT† (Guo et al., Released</td><td></td><td>99.9</td><td>0.063</td><td>0.028</td><td></td></tr><tr><td>2026) MapAnything (Keetha</td><td>Public</td><td>7.0</td><td>0.852</td><td>0.520</td><td>0.211 / 2.096</td></tr><tr><td>et al., 2026) π3 (Wang et al., 2026)</td><td>Public</td><td>8.4</td><td></td><td></td><td></td></tr><tr><td>VGGT (Wang et al., 2025a)</td><td>Public</td><td>5.3</td><td></td><td></td><td></td></tr><tr><td>DUSt3R (Wang et al.,</td><td>Public</td><td>1.2</td><td></td><td></td><td></td></tr><tr><td>2024) MASt3R (Leroy et al., Public 2024)</td><td></td><td>2.2</td><td></td><td></td><td></td></tr></table>

Table 10: Stage-2 runs with and without the aspect-ratio embedding. Both start from the Stage-1 checkpoint and train the Stage-2 configuration (first-generation scenes, full-field-of-view resizing, online camera sampling, 100 epochs, seed 0); the only configuration difference is the embedding. Same protocol as the main tables; 2D3DS pose metrics per tuple.
<table><tr><td></td><td colspan="3">Heterogeneous 2D3DS tuples</td><td colspan="2">Laser tracks, AUC@30</td></tr><tr><td>Stage-2 run</td><td>RRA@30</td><td>RTA@30</td><td>mAA@30</td><td>Mixed</td><td>Panorama</td></tr><tr><td>With the embedding</td><td>91.9</td><td>93.3</td><td>63.8</td><td>65.4</td><td>45.0</td></tr><tr><td>Without the embedding</td><td>90.0</td><td>92.8</td><td>60.6</td><td>61.7</td><td>46.7</td></tr></table>

2D3DS panorama tuples, AUC@30: native 58.1 against 53.9; squeezed to 16:9, 63.8 against 53.9

Table 11: Squeezed input of real panoramas (2D3DS panorama tuples, AUC@30). The rows squeeze every panorama to the stated shape before it enters the pipeline. The panorama detector supplies the content aspect and panorama wrap in the final-model rows. At fixed 16:9 pixels, withholding the aspect-ratio embedding lowers the final model from 78.6 to 42.4. Dashes mark shapes that were not run for that checkpoint.
<table><tr><td>Input shape</td><td>Aspect input</td><td>90-epoch run</td><td>Earlier interpolation</td><td>Final</td></tr><tr><td>Native 2:1</td><td>Content aspect (2.0)</td><td>70.5</td><td>74.9</td><td>75.4</td></tr><tr><td>16:9 squeeze</td><td>Tensor aspect (1.78)</td><td>66.4</td><td>68.8</td><td></td></tr><tr><td>16:9 squeeze</td><td>Content aspect, from the detector (2.0)</td><td>75.9</td><td>78.8</td><td>78.6</td></tr><tr><td>16:9 squeeze</td><td>Embedding withheld</td><td></td><td></td><td>42.4</td></tr><tr><td>4:3 squeeze</td><td>Content aspect, from the detector (2.0)</td><td>71.8</td><td></td><td>76.2</td></tr><tr><td>1:1 squeeze</td><td>Content aspect, from the detector (2.0)</td><td>67.9</td><td>75.1</td><td>76.9</td></tr></table>

Table 12: Real perspective video with centimetre baselines, outside the room-scale tuples of the engine (AUC@30, same tuples, ground truth and scorer for every model). Rotation is solved by every model (RRA@30 of 99–100); the gap is translation direction under micro-baselines. Our row uses the final checkpoint with no view flagged as a panorama. VGGT trains on Replica and ADT.
<table><tr><td>Set (tuples)</td><td>MEOW</td><td>MapAnything</td><td>VGGT</td><td> $\pi ^ { 3 }$ </td><td>DUSt3R</td><td>MASt3R</td></tr><tr><td>Replica (48)</td><td>69.0</td><td>78.4</td><td>93.5</td><td>98.5</td><td>87.1</td><td>98.0</td></tr><tr><td>ADT pinhole (80)</td><td>39.4</td><td>60.8</td><td>53.7</td><td>80.4</td><td>49.3</td><td>83.4</td></tr><tr><td>ADT fisheye (80)</td><td>38.4</td><td>41.0</td><td>42.2</td><td>46.1</td><td>28.4</td><td>34.0</td></tr></table>

Table 13: Efficiency. Top: full pipelines per tuple on one RTX 5090, including image loading, first tuple excluded as warm-up. Bottom: the network forward pass alone at 518 px (random inputs prebuilt on the GPU, 30 iterations after 8 warm-up iterations, CUDA-event timing). A six-point log-log fit gives exponents 1.09 (bf16) and 1.28 (fp32); the local slope rises with N, so this is an empirical scaling and not a complexity statement.
<table><tr><td>Per-tuple wall clock, I/O in- MEOW MapAnything DUSt3R MASt3R cluded (s)</td></tr><tr><td>0.66 0.66 5.54 8.36</td></tr><tr><td>Replica, 8 views (RTX 5090) Laser mixed track, 4 views (RTX 0.48 3.65 4.83 5090)</td></tr><tr><td>Network forward pass, N views 2 4 8 16 24 32</td></tr><tr><td>(RTX 5090)</td></tr><tr><td>BF16 mean (ms) 67.5 126.9 266.8 583.0 965.8 1359.9</td></tr><tr><td>FP32 mean (ms) 129.2 267.7 610.9 1558.4 2867.0 4479.4 Peak memory BF16 (GB) 7.85 9.03 11.42 16.20 20.98 25.77</td></tr><tr><td></td></tr></table>

Table 14: Heterogeneous 2D3DS tuples by tuple size; the final model leads on 79 of the 88 tuples. Per-tuple mAA@30 and per-tuple accuracy at 15<sup>◦</sup>, averaged over the tuples of each size range, for the final model and for Wid3R given the camera type of every view; the last column counts the tuples on which our per-tuple mAA@30 is higher. Training tuples have at most eight views.
<table><tr><td></td><td></td><td colspan="3">MEOW (pixels only)</td><td colspan="3">Wid3R (camera type given)</td><td></td></tr><tr><td>Views</td><td>Tuples</td><td>mAA@30</td><td>RRA@15</td><td>RTA@15</td><td>mAA@30</td><td>RRA@15</td><td>RTA@15</td><td>Ours ahead</td></tr><tr><td>3</td><td>36</td><td>87.5</td><td>99.1</td><td>100.0</td><td>53.6</td><td>91.7</td><td>64.8</td><td>35/36</td></tr><tr><td>4</td><td>13</td><td>85.0</td><td>98.7</td><td>100.0</td><td>56.3</td><td>87.2</td><td>72.4</td><td>13/13</td></tr><tr><td>5-8</td><td>28</td><td>79.7</td><td>92.9</td><td>93.0</td><td>54.0</td><td>88.8</td><td>67.3</td><td>25/28</td></tr><tr><td>9-14</td><td>7</td><td>67.9</td><td>83.9</td><td>81.2</td><td>56.9</td><td>83.6</td><td>70.1</td><td>6/7</td></tr><tr><td>15-24</td><td>4</td><td>29.3</td><td>46.7</td><td>49.2</td><td>52.1</td><td>71.1</td><td>71.2</td><td>0/4</td></tr></table>

![](images/e67c979066f2f5d0933e21b3e5b7e9d8ea82c9462f9cfbd39dd4df49fd64b2bf.jpg)

![](images/1ffe8bafdb04d945c4ab2925b8aeaf17abd070cf203262398557ad7a1cb432ab.jpg)

![](images/9e541d8616425d0253096b8fd7d4776e423a589a86e6ac2b20a0736dd0fd82e4.jpg)

(d) source aspect ratio  
![](images/eafbebff6b502a049790f65b8b96ab78e93f21c7d019108dc1bbd2fff832d941.jpg)

![](images/69911bcfdf74eb5b26009fd878cb1eb261856fa621dd2b2939ad17ad3b8d9351.jpg)

![](images/c4a26a07450469b10da850bd419f69829143466881ea9c6294bbb7bfc1af6687.jpg)  
Figure 6: As-fed distributions of the training stream, read from the per-tuple sampling log of one data-parallel rank of Stage 2 (438,001 tuples): (a) camera-model share, (b) the logged field-of-view parameter, one density strip per camera model (diagonal for rectilinear models, across the image circle for fisheye models, horizontal for spherical crops; full panoramas are 360<sup>◦</sup>), (c) tuple size, (d) source aspect ratio, (e) roll magnitude and tilt, and (f) the principal-point offset magnitude; (e) and (f) use a log density so the 10% uniform tails are visible. The fallback bar in (a) counts views delivered as unaugmented renders (Appendix B).

![](images/349365ab368a44142daf1da0782248492630ebd5365977d13637196b937babfd.jpg)  
Figure 7: What the network receives for one heterogeneous tuple. The public loader picks one bucket from the mean aspect of the tuple, cover-scales and centre-crops every view, so the panorama loses a third of its longitudes and the square views lose a quarter of their height. Our resizing maps every view anisotropically to the same bucket, keeps all content, and passes the aspect ratio and the panorama flag.

![](images/2aaa1215e819a89a029893c349f92e378045b105476a40dca4ac540b7f8ca163.jpg)

![](images/5c25c9a285be9e8f9e779d687c178cf4032018998d205f392a2e29b31bea89d9.jpg)  
(b)  
Figure 8: The laser-scanned benchmark. (a) The merged point cloud of the twelve registered scanner stations, numbered 0–11: a corridor chain and a meeting room. The highlighted stations and link are one mixed tuple of the benchmark (stations 5, 4, 7 and 6); the numbers on the links are the covisibility between the station panoramas along the chain (0.84, 0.68 and 0.87). (b) The four views of that tuple as delivered to the models: a full panorama, a fisheye and two pinhole views resampled from the station scans.

## E DOWNSTREAM PREVIEW: CROSS-CAMERA CORRESPONDENCES

A model that maps the same physical point to the same world coordinate regardless of the camera that observed it yields correspondences without a matcher: for a pixel of view A, its match in view B is the pixel whose predicted world point is the three-dimensional nearest neighbour of A’s predicted point, and a match is kept when it is mutual. No matching head is trained and no descriptor is compared. Figure 9 shows such matches from the final model on real 2D3DS pairs across camera kinds and on pairs of a consumer panorama, a fisheye and a phone photograph of the same desk.

![](images/680757414d3a1decb33ab9bdb292dbb926470442793bda47040a03c46e8a45b9.jpg)  
Figure 9: Cross-camera correspondences of the final model, obtained as mutual three-dimensiona nearest neighbours between predicted world points. Rows one and two: real 2D3DS pairs with ground truth, panorama to fisheye and fisheye to perspective; yellow lines end within $3 ^ { \circ }$ of the true target, blue lines do not. Rows three and four: pairs without ground truth from a consumer panorama, a fisheye and a phone photograph of the same desk, coloured by the predicted threedimensional agreement of the mutual match (brighter is closer).

## F PROTOCOL DETAILS

Pointmaps use accuracy, completeness and normal consistency after one alignment per tuple (the alignment of Umeyama (1991), a least-squares scale and shift, then ICP (Besl & McKay, 1992)). On the two mixed-camera benchmarks our model receives every view of a tuple at the common bucket by full-field-of-view resizing, with the aspect-ratio embedding and the image-detected panorama wrap; on the 2D3DS panorama, Matterport3D, Replica and ADT sets it uses the public crop loader. The open baselines (MapAnything, VGGT, $\pi ^ { 3 }$ , DUSt3R, MASt3R) run with their official loaders and checkpoints on the same tuples, ground truth and scorer. Wid3R runs with its released weights and its official settings, which give the network the camera type of every view; its camera-frame convention was fixed once on a calibration split disjoint from the heterogeneous benchmark (2D3DS areas 1–4) and then frozen.

Pose metrics. Pose metrics are computed over ordered view pairs: RRA@30 and RTA@30 are the fractions of pairs whose relative rotation error and translation-direction error (sign-agnostic, Wang et al., 2023) are below 30<sup>◦</sup>; mAA@30 averages, over thresholds from 1<sup>◦</sup> to 30<sup>◦</sup>, the fraction of pairs with both errors below the threshold, and AUC@30 integrates the same curve; ATE is the trajectory error after one Sim(3) alignment per tuple. On the 2D3DS benchmark the metrics are computed per tuple and averaged over tuples. Pooled over all 4,044 ordered pairs instead, they weight each tuple by its number of pairs, and the four tuples of 15–24 views hold 49% of the pairs (Table 14); pooled mAA@30 / RRA@30 / RTA@30 are 54.2 / 74.5 / 81.0 for MEOW, 54.7 / 88.3 / 84.2 for Wid3R, 18.9 / 41.9 / 58.7 for $\pi ^ { 3 }$ , 14.4 / 47.0 / 49.8 for MapAnything and 9.0 / 32.5 / 44.1 for VGGT. Relative pose metrics need no alignment; ATE and the pointmap metrics are computed after one alignment per tuple. The predicted metric scale is measured without alignment as the ratio of ground-truth to predicted camera-centre distances, median over the view pairs of a tuple and then over tuples: 1.14 on 2D3DS (74% of tuples within ±20%, 91% within ±30%) and 1.26–1.34 on the four laser tracks for the final model, against 0.63 and 0.70 for the public MapAnything weights on 2D3DS and the laser mixed track.

2D3DS panoramas. The panorama tuples of Table 9 apply the sampling that Wid3R describes for 2D3DS (10–30 images per scene, ten draws per scene) to rooms: a random subset of 10–30 panoramas per room (all of them in smaller rooms), ten draws per room over the seven areas, shuffled with seed 0 and truncated to 20; one small room is drawn twice with the same two panoramas, and the repeat is counted once. For single panoramas, the loader scales each panorama to 518 × 259 and crops seven rows, and the ground truth is cropped the same way.

Laser-scanned benchmark. The scene is a multi-room office (a corridor chain and a meeting room) scanned from twelve stations with a Leica BLK360 G2 and registered in Cyclone; each station provides a 1536 × 3072 panorama with along-ray laser depth in the registered frame. The laser benchmark gates consecutive station pairs on the covisibility of the benchmark views themselves, estimated from 1,500 sampled points per direction and symmetrised by the minimum of the two directions (depth agreement within the larger of 10 cm and 3%): 0.25 for the pinhole and fisheye tracks, 0.10 for the panorama track and 0.15 for mixed tuples.

Covisibility envelope. Registration failures concentrate at low pairwise covisibility. On the laser panorama track, the eight tuples on which the final model misses a view pair at 30<sup>◦</sup> (in rotation or translation) have a median minimum pairwise covisibility of 0.042 between their station panoramas (mutual projection of the scans, 10 cm agreement), the sixteen others 0.424, and five of the eight lie below 0.05; camera-sampled training tuples are checked for connectivity at 0.25 (Section 3.2). The envelope subsets of Table 8 add a floor of 0.10 on the benchmark covisibility of every view pair, with consecutive gates of 0.25 on the panorama track (16 tuples) and 0.20 on the mixed track (13 tuples).

Routing and controls. Headline numbers infer the binary full-panorama switch from the pixels; controls that instead use the dataset’s panorama annotation are marked explicitly. Our self-run VGGT and $\pi ^ { 3 }$ rotation accuracies, pooled over pairs as in CAM3R’s public evaluation code and paper, match the values it publishes for them (32.5 against 31.8 and 41.9 against 40.0).

Regenerating the data. The code release will include the commands that regenerate a firstgeneration scene (eight poses, five pinhole, two fisheye and one panoramic render at 64 samples per pixel) and a second-generation scene (panoramas at 3072 × 1536 and 128 samples per pixel), build shards and offline covisibility, draw one covisibility-checked tuple and run ten optimiser steps as a pipeline check; the second-generation split (469/15/15, seed 20260729) is reproduced exactly by a script planned for release, and the optional 935-file CC0 and public-domain asset bundle will include a per-file provenance inventory.