# PhysAg3D:

# Bridging 3D Generative and Agentic Models for High-Fidelity and Physically-Interactive Assets

Bi’an Du<sup>1,2\*</sup> Fanpeng Meng<sup>3\*</sup> Jiemin Fang<sup>3‡</sup> Chen Yang<sup>3</sup> Pufan Li<sup>1,2</sup> Daizong Liu<sup>4</sup> Qi Tian<sup>2,3</sup> Wei Hu<sup>1</sup>

<sup>1</sup>Peking University <sup>2</sup>Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ) <sup>3</sup>Huawei <sup>4</sup>Wuhan University

<sup>\*</sup>Equal Contribution. <sup>‡</sup>Project lead. Corresponding authors.

{scncdba, forhuwei}@pku.edu.cn {fpmeng0610,jaminfong}@gmail.com tian.qi1@huawei.com

## Abstract

Generating interactive 3D assets from a single image requires realistic appearance, fidelity to the depicted object, and plausible physical behavior. Yet detailed geometry alone does not determine how an object should move or deform. We introduce PhysAg3D, which couples 3D generative models with agentic reasoning and tool use. A vision-language agent first infers semantic parts, candidate interactions, and physical priors from the image; a 3D generator then produces a textured whole-object mesh from the same image. A hierarchical interaction representation links these revisable priors to mechanisms and the generated surfaces they control, organizing interaction construction and refinement. Guided by this representation and observations of the mesh, the agent uses geometric and physical tools to construct parts, joints, deformable regions, and surface bindings while retaining useful generated geometry and materials. Rendered interactions and execution measurements guide targeted revisions within a fixed budget. The framework accommodates both rigid articulation and compliant deformation. Quantitative and qualitative evaluations on real photographs and external benchmarks demonstrate detailed, image-aligned assets with plausible interactions. Ablations support bounded refinement and source-surface preservation. Project Page: supra-3d.github.io/PhysAg3D.

![](images/d69601580b3b7cbaa18401a430018a2a622d884af9cc737fd297416714bd9a4a.jpg)  
Figure 1. Detailed appearance with region-specific interaction. Each example shows an input image, its generated asset, and either appearance details or a selected interaction state. (a) A camera shows detailed surface appearance; insets compare corresponding lens-barrel regions visible in the input image and generated asset. (b) A laptop stand adjusts its tray tilt and support-arm elevation. (c) Headphones combine a rigid support with locally compliant ear pads. These development examples show recorded simulated states in (b) and (c); ear-pad deformation uses an approximate compliance model.

## 1 Introduction

Generating interactive 3D assets from a single image offers a scalable way to populate virtual environments and embodied simulations with objects from the real world (Mildenhall et al., 2020; Kerbl et al., 2023; Mo et al., 2019; Xia et al., 2020; Xiang et al., 2020). A useful asset should capture the depicted instance, retain realistic geometric and material detail, and respond to actions in ways appropriate to the object. These goals concern different aspects of quality: a visually convincing reconstruction can remain static, while a functional mechanism can belong to an asset that no longer resembles its reference. The task therefore requires bringing appearance, instance alignment, and physical interaction into a coherent realization.

Recent advances provide strong foundations for this task. Image-conditioned generators synthesize detailed geometry and materials (Poole et al., 2023; Jun & Nichol, 2023; Nichol et al., 2022; Lin et al., 2023; Qian et al., 2023; Liu et al., 2023a; Shi et al., 2023; Liu et al., 2023b; Hong et al., 2024; Long et al., 2024; Tochil et al., 2024; Xu et al., 2024; Du et al., 2024a; Voleti et al., 2024; Tang et al., 2024; Xiang et al., 2025b;a); articulated and physical generation methods infer part structure, motion, and material behavior (Mo et al., 2019; Xia et al., 2020; Yu et al., 2023; Wang et al., 2019; Xiang et al., 2020; Liu et al., 2025; 2026; Cao et al., 2026; Makoviychuk et al., 2021; Hu et al., 2020); and agentic systems construct and revise assets through executable programs, visual grounding, and feedback (Ahn et al., 2022; Li et al., 2023a; Huang et al., 2023b;a; Driess et al., 2023; Brohan et al., 2023; Singh et al., 2023; Valmeekam et al., 2023; Le et al., 2025; Zhou et al., 2026; Wang et al., 2026). A remaining challenge is to maintain agreement between what an object is expected to do, how that behavior is implemented, and which visible surfaces actually respond. A single image underdetermines hidden mechanisms and physical parameters, and a generated mesh need not expose the semantic parts required for interaction. Furthermore, a valid joint or deformable simulation does not ensure that the corresponding detailed surface moves with it. Addressing these discrepancies can require changes to geometry and physical structure, creating a further tension between functional construction and preservation of the depicted appearance.

We introduce PhysAg3D, a framework for single-image interactive asset generation through generative–agentic coupling. Our central idea is to combine the instance-specific geometry and appearance supplied by a pretrained 3D generator with an agent’s ability to reason about object behavior and construct executable physical models. The generated asset provides a detailed visual foundation; agentic reasoning turns this foundation into an interactive realization and revises it in response to observed behavior. This division of capabilities allows physical construction to make use of rich generated surfaces while remaining responsive to the structure and interactions of the depicted object. Figure 1 illustrates this goal through mixer-head articulation, compass-lid closure, telescope reorientation, and local ear-pad deformation. The examples associate actions with specific moving or deforming regions while retaining recognizable object structure and surface detail.

Specifically, a vision-language agent first infers revisable semantic and physical priors from the image, and a 3D generator produces a textured whole-object mesh from the same image. We propose a hierarchical interaction representation for this coupling, connecting semantic requirements, physical mechanisms, and concrete asset realizations. Explicit correspondences associate intended interactions with the structures and visible surfaces that implement them. Guided by these relationships and observations of the mesh, the agent uses geometric and physical tools to construct parts, joints or compliant regions, and surface bindings while retaining useful generated geometry and materials. Rendered interactions and execution measurements then guide targeted revisions with bounded construction and validation attempts. The representation expresses both rigid articulation and compliant deformation through their respective relationships between physical state and visible appearance. Our contributions are threefold:

• A generative–agentic framework that grounds interaction construction in image-derived priors while retaining useful surfaces and materials from detailed whole-object generation.

• A hierarchical interaction representation that connects requirements, mechanisms, and concrete surface realizations to organize construction and feedback-guided refinement.

• Quantitative and qualitative evaluations on real photographs and external benchmarks, assessing appearance, image alignment, and interaction attributes, with ablations of bounded refinement and source-surface preservation.

## 2 Related Work

Image-conditioned 3D asset generation. Early text-to-3D systems distilled diffusion priors into 3D representations (Poole et al., 2023; Jun & Nichol, 2023; Nichol et al., 2022; Lin et al., 2023; Qian et al., 2023; Tang et al., 2023; Du et al., 2026); feed-forward and multi-view methods improved single-image consistency and speed (Liu et al., 2023a; Shi et al., 2023; Liu et al., 2023b; Hong et al., 2024; Long et al., 2024; Du et al., 2024b; Tochil et al., 2024; Xu et al., 2024; Voleti et al., 2024; Tang et al., 2024). TRELLIS introduces structured 3D latents, and TRELLIS.2 supports detailed geometry and physically based materials (Xiang et al., 2025b;a). UniLat3D unifies geometry and appearance in a shared latent space for single-stage image-to-3D generation (Wu et al., 2026). SINGAPO conditions articulated parts on images and part relationships (Liu et al., 2025); PAct jointly generates geometry, composition, and articulation (Liu et al., 2026); PhysX-Omni covers rigid, deformable, and articulated objects (Cao et al., 2026). Shape, surface, and texture reconstruction further improve appearance (Gao et al., 2022; Li et al., 2023b; Huang et al., 2023c; Cao et al., 2023; Chen et al., 2024; Guo et al., 2020; Müller et al., 2022; Laine et al., 2020). More recently, SUPRA-TEX generates relightable materials faithful to reference images by anchoring visible-surface observations in material latent space while completing unobserved regions (Yang et al., 2026). PhysAg3D studies agent-driven construction and revision around a detailed whole-object reconstruction, linking image-derived requirements to mechanisms and visible surfaces.

Articulation and physical modeling of 3D assets. PartNet, PartNet-Mobility, GAPartNet, and Shape2Motion support research on parts, affordances, and mobility (Mo et al., 2019; Xia et al., 2020; Yu et al., 2023; Wang et al., 2019). Articulate AnyMesh segments movable parts and refines joints for open-vocabulary meshes (Qiu et al., 2025). Particulate predicts segmentation, kinematics, and motion constraints in one pass (Li et al., 2025); Instruct-Particulate adds semantic and point-prompt conditioning, combining 3D generation with vision-language inference for image-to-asset articulation (Li et al., 2026b). PhysAg3D focuses on connecting intended interactions, physical entities, exported bindings, and observed surface responses, including compliant regions whose simulation and visible geometry differ.

Agentic 3D modeling and refinement. Articulate-Anything uses vision-language reasoning and actor–critic refinement for articulated modeling (Le et al., 2025). Language-guided robotics connects instructions to executable actions and closed-loop spatial reasoning (Ahn et al., 2022; Li et al., 2023a; Huang et al., 2023b;a; Driess et al., 2023; Brohan et al., 2023), while planning studies emphasize replanning and verification (Singh et al., 2023; Valmeekam et al., 2023). ArtiCraft provides a part-and-joint SDK, object-specific tests, and structured execution feedback (Zhou et al., 2026). aDSL unifies hierarchical modeling and spatial relations for planning, execution, and critique, combining procedural scaffolds with TRELLIS through SpaceControl (Wang et al., 2026). Shape programs and CAD languages offer related compositional abstractions and persistent references (Jones et al., 2020; 2025; Cascaval et al., 2023). PhysAg3D links requirements, mechanisms, and concrete realizations to generated surfaces. Visual and execution evidence guides local revision while preserving useful geometry.

Simulation and visual grounding substrates. MuJoCo, PyBullet, Isaac Gym, and differentiable simulators provide complementary dynamics and contact substrates (Todorov et al., 2012; Coumans & Bai, 2016; Makoviychuk et al., 2021; Hu et al., 2020). Segment Anything, DINOv2, and CLIP provide visual grouping and image–text alignment primitives (Kirillov et al., 2023; Oquab et al., 2024; Radford et al., 2021). These form contextual foundations for PhysAg3D’s explicit interaction-to-surface representation.

## 3 Method

## 3.1 Task Definition and Framework Overview

Given a single image I, PhysAg3D generates a textured 3D asset with object-specific interactions. An asset contains visible geometry and materials, physical entities and parameters, controls, and bindings from physical state to visible surfaces. We evaluate appearance, image alignment, and physical behavior separately; construction targets the declared interactions, since hidden structure, scale, and material parameters are not determined by one image.

Figure 2 summarizes the workflow. Image interpretation produces revisable semantic and physical hypotheses, while a pretrained generator provides a detailed whole-object mesh. Mechanism reasoning connects these sources through interaction intent S (what should happen), physical structure K (which mechanism produces it), and concrete realization X (the implementing asset entities and bindings). Asset construction realizes these mechanisms and bindings; evidence attribution links observations to the responsible regions for local revision. These are functional roles in one agentic loop. Appendix A details the interaction representation, construction procedures, validation checks, and candidate-selection policy.

## 3.2 Generative–Agentic Coupling

Image priors and generated geometry. A vision-language agent infers provisional hypotheses $\Pi _ { 0 } = \Phi ( I )$ about parts, materials, connections, candidate interactions, and scale, while retaining ambiguities. A pretrained generator produces a textured whole-object mesh $M _ { 0 } = G ( I ) ;$ ; our implementation uses TRELLIS.2 (Xiang et al., 2025a). The agent inspects rendered views and geometry of $M _ { 0 }$ before fixing partitions or mechanisms. Priors guide construction but do not condition G, so the coupling occurs through interpretation, construction, and feedback without joint training.

(1) Geometric and Physical Priors  
(2) Generative-agentic coupling  
(3) Interactive Asset Synthesis  
![](images/69b847247d249626225da83d74aca2dbdd2dfc3280679e4bcb5830079f8684c7.jpg)  
Figure 2. Overview of PhysAg3D with an excavator example. (1) Image-derived priors are combined with a generated textured mesh. (2) Interaction intent $S$ specifies lifting, reaching, and curling; physical structure K specifies the corresponding joints; concrete realization X implements them with asset entities and bindings, illustrated here by its visible surfaces. (3) Asset construction binds surfaces to physical entities, and evidence attribution links observations back to the representation for local revision. Mechanism reasoning, asset construction, and evidence attribution operate within one agentic loop. Close-ups show body–boom, boom–stick, and stick–bucket connections; reference and posed states illustrate articulation.

Hierarchical interaction representation. For candidate version $v ,$ we define

$$
\mathcal { R } _ { v } = ( S _ { v } , K _ { v } , X _ { v } , \Gamma _ { v } , \rho _ { v } ) , \qquad \mathcal { H } _ { v } = ( \mathcal { R } _ { v } , E _ { v } ) ,\tag{1}
$$

where S, K, and X encode semantic requirements, mechanisms, and concrete realizations; Γ links their entities; $\rho$ records correspondence to visible regions of $M _ { 0 } ;$ and E stores candidate observations. The revision index v is distinct from execution time t.

A requirement is

$$
r = ( U _ { r } , a _ { r } , \mathcal { O } _ { r } , \psi _ { r } , \mathcal { D } _ { r } ) ,\tag{2}
$$

where $U _ { r }$ is the target region, $a _ { r }$ the action protocol, $\mathcal { O } _ { r }$ the observation, $\psi _ { r }$ the response test, and $\mathcal { D } _ { r }$ the initial conditions and scope. Thus, lid opening is measured relative to its support, while local compliance is measured after removing common rigid motion. Material compliance and articulation are separate attributes.

The mechanism level K specifies rigid or compliant domains, supports, connections, degrees of freedom, joint frames, limits, and material assumptions. The realization level X identifies actual meshes, materials, scene nodes, controls, simulation entities, and exported bindings, while retaining construction or loading failures. $\mathbf { A } \mathbf { n }$ intended mechanism can therefore remain valid even when its realization is incomplete. In Figure 2, “Interaction intent” denotes $S ,$ “Physical structure” denotes $K .$ , and “Visible surfaces” depicts the surface component of X, whose full scope includes the implementing entities and bindings.

Correspondences across levels and surfaces. The relations Γ connect requirements to mechanisms, mechanisms to realized entities, semantic parts to visible regions, and physical drivers to bindings. The source map $\rho$ records retained, modified, added, deleted, and unresolved surface/material regions relative to $M _ { 0 }$ . In Figure 2, the inter-level links illustrate Γ, the native surface and material anchors illustrate $\rho ,$ and attributed observations contribute to $E .$ These maps distinguish intended relationships from exported ones: an intended hinged panel may be found attached to the base. Topology changes require an updated correspondence rather than reuse of obsolete face indices. The hierarchy expresses progressively stronger commitments about one candidate, not a fixed sequence of agent roles; requirements can select mechanisms, and observations can retrieve the associated entities for inspection.

## 3.3 Physical Construction with Appearance Preservation

The agent grounds semantic parts and compliant domains in image evidence, rendered views, and measurements of $M _ { 0 }$ Connected geometry can contain multiple moving parts, and texture boundaries need not be mechanical boundaries, so every actionable surface selection is recorded in a declared frame. Construction uses Blender for geometry/material edits and physical assembly tools for simulation entities. Rigid construction sets body ownership, parent–child connections, joint frames and limits, collision geometry, and controls. Compliant construction sets a physical discretization, material assumptions, supports, loads, and a map to the visible skin. Both preserve useful source geometry, UVs, and materials while recording the purpose and source correspondence of bounded edits.

Table 1. Single-image interactive generation on 50 in-the-wild objects with 110 predefined actions. Image metrics use camera-grid-aligned views. ACR, MMR, and RAR are object-level macro averages; FVSP requires all actions to complete without video-visible instability. Art./Def.: articulated/deformable support; ✓/×: reported or evaluated/no reported support. Best and second-best values are bold and italic, respectively.
<table><tr><td>Method</td><td colspan="2">Object Type Support</td><td colspan="3">Input Consistency</td><td colspan="2">Action Quality</td><td>Range</td><td>Stability</td></tr><tr><td></td><td>Art.</td><td>Def.</td><td>FG-IoU↑</td><td>PSNR↑</td><td>CLIP↑</td><td>ACR↑</td><td>MMR↓</td><td>RAR↑</td><td>FVSP↑</td></tr><tr><td>Particulate</td><td>√</td><td>X</td><td>0.607</td><td>14.24</td><td>0.853</td><td>46.7%</td><td>57.5%</td><td>34.7%</td><td>24.0%</td></tr><tr><td>PAct</td><td>√</td><td>X</td><td>0.531</td><td>12.53</td><td>0.806</td><td>35.0%</td><td>87.2%</td><td>18.0%</td><td>16.0%</td></tr><tr><td>ArtiCraft</td><td>√</td><td>X</td><td>0.514</td><td>12.25</td><td>0.768</td><td>56.8%</td><td>51.8%</td><td>34.3%</td><td>28.0%</td></tr><tr><td>PhysX-Omni</td><td>√</td><td>√</td><td>0.504</td><td>13.06</td><td>0.817</td><td>41.2%</td><td>68.5%</td><td>26.0%</td><td>16.0%</td></tr><tr><td>aDSL</td><td>√</td><td>X</td><td>0.520</td><td>12.37</td><td>0.779</td><td>66.8%</td><td>40.2%</td><td>48.8%</td><td>42.0%</td></tr><tr><td>PhysAg3D (ours)</td><td>√</td><td>√</td><td>0.835</td><td>16.47</td><td>0.892</td><td>91.3%</td><td>22.8%</td><td>91.3%</td><td>88.0%</td></tr></table>

For an executable realization $X$

$$
\begin{array} { r } { z _ { t + 1 } = F _ { X } ( z _ { t } , u _ { t } ) , \qquad V _ { t } = B _ { X } ( z _ { t } ; V _ { \mathrm { r e s t } } ) , \qquad y _ { r , t } = \mathcal { O } _ { r , \Gamma , \rho } ( V _ { t } , z _ { t } ) . } \end{array}\tag{3}
$$

Here $F _ { X }$ is the implemented transition, $B _ { X }$ the exported binding, and $y _ { r , t }$ the observation used to assess r. A kinematic playback and a physics simulator may produce similar visible states, but only the latter supplies dynamics evidence. Rigid vertices follow their assigned body transform; for a compliant region, one valid binding is

$$
V _ { t } = V _ { \mathrm { { r e s t } } } + W ( P _ { t } - P _ { \mathrm { { r e s t } } } ) ,\tag{4}
$$

where $P _ { t }$ are physical-node positions and W maps their displacements to visible vertices in a shared frame. Binding checks verify node identities, dimensions, weights, and rest geometry against the export. A correct rest pose or moving proxy alone does not establish the intended surface response.

## 3.4 Refinement with Visual and Physical Feedback

Each candidate is evaluated with complementary evidence: renders assess image alignment, retained appearance, part separation, and visible response; source checks assess preserved surfaces and materials; execution checks assess actions, ranges, contacts, supports, and applicable deformation or recovery. MuJoCo (Todorov et al., 2012) is used for supported physical models. A moving proxy, a joint reaching a target, and the intended visible surface responding are distinct observations. Each $E _ { v }$ record stores the requirement, target entities, candidate identity, action, initial state, runtime, and assessment configuration, with outcomes marked unassessed, pass, fail, or inconclusive. Errors without trajectories provide no measured response, and earlier dynamic evidence is invalidated after a build change.

For an unmet requirement, feedback combines the discrepancy or missing evidence with the related mechanisms, realized entities, and bindings obtained through $\Gamma _ { v }$ . Given remaining allowances $b _ { v }$ , the agent proposes

$$
\begin{array} { r } { \begin{array} { c c } { o _ { v } \sim \pi ( I , \mathcal { R } _ { v } , E _ { v } , b _ { v } ) , \qquad X _ { v + 1 } = \mathrm { A p p l y } ( X _ { v } , o _ { v } ) , } \\ { \mathcal { R } _ { v + 1 } = \mathrm { R e c o n c i l e } ( \mathcal { R } _ { v } , o _ { v } , X _ { v + 1 } ; M _ { 0 } ) . } \end{array} } \end{array}\tag{5}
$$

Correspondences localize inspection but do not guarantee a causal diagnosis: failures may arise from partitions, frames, supports, controls, material assumptions, bindings, or observation definitions. Reconciliation records the actual export and keeps unresolved discrepancies; accepted requirements remain traceable across revisions, so a reduced motion range cannot silently satisfy a full-closure requirement.

A declared policy bounds construction and validation and preserves prior candidates when revisions fail. The system returns the best eligible candidate under a fixed evidence-based selection rule, together with its interactive preview and unresolved requirements. A preview may be useful without a valid simulation trajectory, while its dynamics remain unverified. Appendix A gives the implementation budgets, checks, and selection policy.

## 4 Experiments

## 4.1 Experimental Settings

We evaluate single-image interactive generation on real photographs and external benchmarks, and ablate bounded refinement and source-surface preservation. Implementation and scoring details appear in Appendices A and B.

![](images/ae07479bff7747dc4fc9e38696c47b58d498f00088dd3deb6209da4b1e7ca80b.jpg)  
Figure 3. Qualitative comparison on real photographs. Printer and chair panels compare appearance and displayed motion across aDSL, PhysX-Omni, and PhysAg3D. The right column shows initial, pressed, and released states of a deformable asset.

Common protocol. All comparisons fix input lists, method versions, and inference configurations; image-conditioned methods receive the same source image. We record auxiliary conditioning, output support, and computational cost. Candidate selection uses only within-run evidence, without benchmark annotations or final evaluation scores. Refinement permits at most two Builder repairs after initial construction and returns the best available candidate. All attempted inputs remain in the outcome accounting. Static assets are evaluated for input consistency; missing or invalid interaction videos count as failures within the fixed action denominator.

Qualitative development examples. Figure 1 shows separate development examples: mixer and compass articulation, approximate compliant ear pads, and telescope poses specified by joint parameters without dynamic validation.

## 4.2 Single-Image Interactive Generation in the Wild

Benchmark and comparisons. Our benchmark contains more than 200 real-object photographs; evaluation uses a fixed subset of 50 objects with 110 predefined semantic actions, spanning rigid, compliant, and mixed objects. Development cases are identified separately, and method-specific preprocessing is documented. We compare PhysAg3D with Particulate, PhysX-Omni, PAct, ArtiCraft, and aDSL (Li et al., 2025; Cao et al., 2026; Liu et al., 2026; Zhou et al., 2026; Wang et al., 2026). Additional qualitative comparisons appear in Appendix C.

Evaluation metrics. For input-view consistency, a fixed spatial grid search selects the rendered camera view closest to the input; foreground IoU, PSNR, and CLIP similarity are computed on the aligned pair. Action Completion Rate (ACR) measures correct semantic completion. Strict Mis-Motion Rate (MMR) counts missing, incomplete, incorrectly related, structurally broken, or spuriously coupled actions. Video-based Range Achievement Rate (RAR) scores the maximum valid visible range in {0, 0.25, 0.5, 0.75, 1} after the ACR semantic gate. These three metrics are macro-averaged over objects. Function-and-Visible-Stability Pass (FVSP) is the fraction of objects completing all expected actions without video-visible instability. It assesses rendered video evidence, not simulator-log stability.

PhysAg3D leads all reported input-consistency and interaction metrics in Table 1, reaching 91.3% ACR and 88.0% FVSP versus 66.8% and 42.0% for aDSL, the strongest baseline on these two measures. Figure 3 compares appearance and visible interaction states under consistent rendering conditions.

Table 2. Reference-based comparison on PhysXVerse and PhysX-Mobility. CD is in $1 0 ^ { - 3 }$ and F-score in $1 0 ^ { - 2 }$ at threshold 0.05. Mat., Aff., Kin., and Desc. denote material, affordance, kinematics, and description. Bold and underlined values indicate the best and second-best scores. A dash indicates an entry with no reported score.
<table><tr><td>Method</td><td>PSNR↑</td><td>CD↓</td><td>F-score↑</td><td>Scale↓</td><td>Mat.↑</td><td>Aff.↑</td><td>Kin.↑</td><td>Desc.↑</td></tr><tr><td colspan="9">Dataset: PhysXVerse</td></tr><tr><td>Articulate-Anything (Le et al., 2025)</td><td>14.03</td><td>48.77</td><td>46.44</td><td></td><td></td><td></td><td>0.30</td><td></td></tr><tr><td>MonoArt (Li et al., 2026a)</td><td>19.68</td><td>7.03</td><td>85.27</td><td></td><td></td><td></td><td>0.38</td><td></td></tr><tr><td>PhysXGen (Cao et al., 2025a)</td><td>19.41</td><td>15.19</td><td>83.56</td><td>309.31</td><td>16.51</td><td>9.40</td><td>0.35</td><td>11.84</td></tr><tr><td>PhysX-Anything (Cao et al., 2025b)</td><td>15.97</td><td>37.06</td><td>40.46</td><td>298.19</td><td>15.65</td><td>10.50</td><td>0.42</td><td>21.38</td></tr><tr><td>PhysX-Omni (Cao et al., 2026)</td><td>21.52</td><td>2.95</td><td>91.28</td><td>2.79</td><td>27.23</td><td>21.47</td><td>0.92</td><td>31.05</td></tr><tr><td>PAct (Liu et al., 2026)</td><td>21.10</td><td>3.20</td><td>90.90</td><td>3.05</td><td>26.50</td><td>20.80</td><td>0.92</td><td>30.30</td></tr><tr><td>ArtiCraft (Zhou et al., 2026)</td><td>16.24</td><td>12.46</td><td>77.08</td><td>5.07</td><td>20.94</td><td>27.13</td><td>0.94</td><td>23.08</td></tr><tr><td>aDSL (Wang et al., 2026)</td><td>17.18</td><td>9.76</td><td>80.93</td><td>4.16</td><td>21.87</td><td>26.14</td><td>0.93</td><td>24.92</td></tr><tr><td>PhysAg3D (ours)</td><td>24.26</td><td>2.08</td><td>94.34</td><td>2.17</td><td>32.18</td><td>28.24</td><td>0.96</td><td>38.16</td></tr><tr><td colspan="9">Dataset: PhysX-Mobility</td></tr><tr><td>Articulate-Anything (Le et al., 2025)</td><td>15.02</td><td>16.09</td><td>66.95</td><td></td><td></td><td></td><td>0.64</td><td></td></tr><tr><td>MonoArt (Li et al., 2026a)</td><td>16.46</td><td>6.35</td><td>87.41</td><td></td><td></td><td></td><td>0.44</td><td></td></tr><tr><td>PhysXGen (Cao et al., 2025a)</td><td>15.75</td><td>35.32</td><td>79.62</td><td>46.58</td><td>16.02</td><td>8.73</td><td>0.39</td><td>11.60</td></tr><tr><td>PhysX-Anything (Cao et al., 2025b)</td><td>16.57</td><td>23.13</td><td>89.51</td><td>22.58</td><td>22.58</td><td>16.29</td><td>0.79</td><td>26.28</td></tr><tr><td>PhysX-Omni (Cao et al., 2026)</td><td>18.38</td><td>4.70</td><td>88.50</td><td>2.78</td><td>24.09</td><td>16.58</td><td>0.86</td><td>28.40</td></tr><tr><td>PAct (Liu et al., 2026)</td><td>18.00</td><td>5.20</td><td>86.50</td><td>3.10</td><td>23.00</td><td>16.00</td><td>0.86</td><td>27.00</td></tr><tr><td>ArtiCraft (Zhou et al., 2026)</td><td>15.36</td><td>10.74</td><td>78.46</td><td>5.26</td><td>17.62</td><td>20.37</td><td>0.88</td><td>22.14</td></tr><tr><td>aDSL (Wang et al., 2026)</td><td>16.08</td><td>8.84</td><td>81.93</td><td>4.64</td><td>19.18</td><td>19.62</td><td>0.87</td><td>23.86</td></tr><tr><td>PhysAg3D (ours)</td><td>21.14</td><td>2.94</td><td>92.56</td><td>2.36</td><td>29.18</td><td>23.16</td><td>0.93</td><td>34.58</td></tr></table>

Table 3. PhysX-Bench comparison. RQS: visual quality; MCS: 3D consistency; DQS: absolute scale; APS, KPS, MPS, and DCS denote affordance, kinematic, material, and description scores. All columns are higher-is-better. Bold and underlined values indicate the best and second-best scores, including ties. A dash or NR denotes an unreported score.
<table><tr><td>Method</td><td>CLIP</td><td>RQS</td><td>MCS</td><td>DQS</td><td>APS</td><td>KPS</td><td>MPS</td><td>DCS</td></tr><tr><td>Articulate-Anything (Le et al., 2025)</td><td>0.55</td><td>88.46</td><td>55.27</td><td></td><td></td><td>71.25</td><td></td><td></td></tr><tr><td>MonoArt (Li et al., 2026a)</td><td>0.84</td><td>96.20</td><td>82.56</td><td></td><td></td><td>68.32</td><td></td><td></td></tr><tr><td>PhysXGen (Cao et al., 2025a)</td><td>0.80</td><td>85.93</td><td>73.50</td><td>24.21</td><td>66.07</td><td>69.17</td><td>NR</td><td>22.24</td></tr><tr><td>PhysX-Anything (Cao et al., 2025b)</td><td>0.55</td><td>70.81</td><td>52.71</td><td>50.20</td><td>59.96</td><td>65.99</td><td>44.70</td><td>26.89</td></tr><tr><td>PhysX-Omni (Cao et al., 2026)</td><td>0.77</td><td>90.00</td><td>64.48</td><td>64.26</td><td>70.57</td><td>80.72</td><td>59.89</td><td>39.02</td></tr><tr><td>PAct (Liu et al., 2026)</td><td>0.74</td><td>88.50</td><td>62.00</td><td>61.50</td><td>68.00</td><td>82.00</td><td>58.50</td><td>36.50</td></tr><tr><td>ArtiCraft (Zhou et al., 2026)</td><td>0.49</td><td>64.18</td><td>46.12</td><td>56.24</td><td>74.16</td><td>85.88</td><td>66.24</td><td>24.18</td></tr><tr><td>aDSL (Wang et al., 2026)</td><td>0.53</td><td>68.24</td><td>50.16</td><td>60.08</td><td>74.82</td><td>84.13</td><td>67.18</td><td>26.86</td></tr><tr><td>PhysAg3D (ours)</td><td>0.84</td><td>95.18</td><td>75.24</td><td>72.16</td><td>82.34</td><td>88.18</td><td>72.26</td><td>48.12</td></tr></table>

## 4.3 Evaluation on External Benchmarks

Reference-based evaluation. PhysXVerse and PhysX-Mobility provide reference assets for appearance, geometry, and attribute evaluation, with the dataset split reported by PhysX-Omni (Cao et al., 2026). In Table 2, PhysAg3D achieves the best reported score in every column on both datasets, improving PSNR over the strongest baseline by 2.74 and 2.76 dB, respectively.

Reference-free evaluation. PhysX-Bench evaluates generated assets using image and behavior evidence (Cao et al., 2026). We use its released evidence-generation and scoring pipeline with fixed judge configuration and prompts. In Table 3, PhysAg3D ties MonoArt on CLIP, ranks second on RQS and MCS, and leads DQS, APS, KPS, MPS, and DCS. These results show strong attribute scores alongside competitive visual quality. The articulation score uses joint playback and does not establish dynamic contact stability.

Qualitative comparison on PhysXVerse. Figure 4 compares five selected inputs with PAct, Particulate, ArtiCraft, and aDSL (Liu et al., 2026; Li et al., 2025; Zhou et al., 2026; Wang et al., 2026). PhysAg3D preserves patterned headphones during earcup rotation, camera housing during lens extension, and ornate cart surfaces during wheel rotation; the microscope and butterfly show localized mechanism and wing motion. ArtiCraft and aDSL simplify fine geometry or patterns, while some PAct and Particulate outputs show fragmented surfaces, misplaced parts, or implausible motion.

## 4.4 Human Perceptual Evaluation

We evaluate six methods for input-image consistency and motion plausibility. First, second, third, and last place receive 5, 4, 3, and 1 points; unranked methods receive 2. First and last place are required, whereas second and third are optional. A separate multi-select question records visible motion collapse.

Across 102 valid evaluations (Table 4), PhysAg3D leads in mean input-consistency (4.63 versus 2.98) and motionplausibility scores (4.38 versus 2.84), and first-place selections. Collapse is flagged 7 times versus 11–47 for baselines. These judgments measure perceived fidelity and motion plausibility, not calibrated dynamics or contact stability.

![](images/a78736688abab84007dbbc688bbf23b9e1d1cb069c8de203fab0263d34e89931.jpg)  
Figure 4. Qualitative comparison on PhysXVerse. Each row shows the input and two states per method, with fixed viewpoint and scale within each pair. Red arrows and arcs indicate translation or rotation in our results. The states show appearance and articulation playback, not calibrated dynamics or contact stability

Table 4. Human perceptual evaluation over 102 valid evaluations. “#1” denotes first-place selections. “Collapse” counts evaluations in which visible motion collapse was flagged; multiple methods may be selected. Best and second-best results are bolded and underlined, respectively.
<table><tr><td></td><td colspan="2">Input-image consistency</td><td colspan="2">Motion plausibility</td><td></td></tr><tr><td>Method</td><td>Mean↑</td><td>#1↑</td><td>Mean↑</td><td>#1↑</td><td>Collapse↓</td></tr><tr><td>Particulate</td><td>2.98</td><td>7</td><td>2.84</td><td>14</td><td>21</td></tr><tr><td>PAct</td><td>1.99</td><td>1</td><td>1.73</td><td>0</td><td>47</td></tr><tr><td>ArtiCraft</td><td>2.01</td><td>1</td><td>2.06</td><td>1</td><td>28</td></tr><tr><td>PhysX-Omni</td><td>2.03</td><td>5</td><td>2.24</td><td>7</td><td>15</td></tr><tr><td>aDSL</td><td>2.38</td><td>4</td><td>2.75</td><td>11</td><td>11</td></tr><tr><td>PhysAg3D (ours)</td><td>4.63</td><td>84</td><td>4.38</td><td>69</td><td>7</td></tr></table>

## 4.5 Ablation Study

Ablation settings. The three settings share PhysX-Bench inputs, cached TRELLIS.2 reconstructions, image priors, model and decoding settings, construction tools, and final scoring. Full pipeline preserves useful source geometry, UVs, and materials while allowing evidence-guided edits and up to two Builder repairs using visual and physical feedback. One-shot construction returns the same initial build without repair, but still undergoes final evaluation. w/o surface preservation removes the construction-time geometry, UV, and material preservation constraints while retaining the initial reconstruction, physical-to-visible bindings, repair budget, and candidate-selection rule.

Table 5. Ablation on PhysX-Bench.
<table><tr><td>Setting</td><td>CLIP↑</td><td>RQS↑</td><td>KPS↑</td><td>MPS↑</td></tr><tr><td>w/o surface preservation</td><td>0.68</td><td>77.50</td><td>75.00</td><td>59.00</td></tr><tr><td>One-shot construction</td><td>0.72</td><td>82.80</td><td>68.50</td><td>55.00</td></tr><tr><td>Full pipeline</td><td>0.84</td><td>95.18</td><td>88.18</td><td>72.26</td></tr></table>

Results. Table 5 shows that Full pipeline improves KPS and MPS over One-shot by 19.68 and 17.26 points. Removing surface preservation reduces CLIP by 0.16 and RQS by 17.68. These results support bounded refinement and source preservation; failed and inconclusive runs remain in the evaluation.

## 5 Conclusion

We presented PhysAg3D for generating interactive 3D assets from a single image. Its hierarchical representation connects interaction requirements, physical mechanisms, and visible surfaces to guide construction and feedback-based refinement. Experiments support appearance preservation and plausible interactions, with ablations indicating benefits from refinement and surface preservation. Recovering hidden structure and material properties remains a limitation.

## Acknowledgments

This research was financially supported by the Open Research Fund from Guangdong Laboratory of Artificial Intelligence and Digital Economy (SZ), under Grant No. GML-KF-26-29.

## References

Michael Ahn, Anthony Brohan, Noah Brown, et al. Do as i can, not as i say: Grounding language in robotic affordances. arXiv preprint arXiv:2204.01691, 2022. URL https://arxiv.org/abs/2204.01691.

Anthony Brohan, Noah Brown, Justice Carbajal, et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. arXiv preprint arXiv:2307.15818, 2023. URL https://arxiv.org/abs/2307.15818.

Chenjie Cao, Nikolaos Sarafianos, Tony Tung, et al. TexFusion: Synthesizing 3d textures with text-guided image diffusion models. arXiv preprint arXiv:2310.13766, 2023. URL https://arxiv.org/abs/2310.13766.

Ziang Cao, Zhaoxi Chen, Liang Pan, and Ziwei Liu. PhysX-3D: Physical-grounded 3D asset generation. arXiv preprint arXiv:2507.12465, 2025a. URL https://arxiv.org/abs/2507.12465.

Ziang Cao, Fangzhou Hong, Zhaoxi Chen, Liang Pan, and Ziwei Liu. PhysX-Anything: Simulation-ready physical 3D assets from single image. arXiv preprint arXiv:2511.13648, 2025b. URL https://arxiv.org/abs/2511.13648.

Ziang Cao, Yinghao Liu, Haitian Li, Runmao Yao, Fangzhou Hong, Zhaoxi Chen, Liang Pan, and Ziwei Liu. PhysX-Omni: Unified simulation-ready physical 3D generation for rigid, deformable, and articulated objects. arXiv preprint arXiv:2605.21572, 2026. URL https://arxiv.org/abs/2605.21572.

Dan Cascaval, Rastislav Bodik, and Adriana Schulz. A lineage-based referencing DSL for computer-aided design. Proceedings ofthe ACM on Programming Languages, 7(PLDI):109:1–109:24, 2023. doi: 10.1145/3591223. URL https://doi.org/10.1145/3591223.

Rui Chen et al. GenesisTex: Adapting image denoising diffusion to texture space. arXiv preprint arXiv:2403.17782, 2024. URL https://arxiv.org/abs/2403.17782.

Erwin Coumans and Yunfei Bai. Pybullet, a python module for physics simulation for games, robotics and machine learning. Technical report, 2016. URL https://pybullet.org.

Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, et al. PaLM-E: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023. URL https://arxiv.org/abs/2303.03378.

Bi’an Du, Xiang Gao, Wei Hu, and Renjie Liao. Generative 3d part assembly via part-whole-hierarchy message passing. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20850–20859. IEEE, 2024a.

Bi’an Du, Wei Hu, and Renjie Liao. Multi-scale latent point consistency models for 3d shape generation. arXiv preprint arXiv:2412.19413, 2024b.

Bi’an Du, Daizong Liu, Pufan Li, and Wei Hu. From part to whole: 3d generative world model with an adaptive structural hierarchy. arXiv preprint arXiv:2603.21557, 2026.

Jun Gao, Tianchang Shen, Zian Wang, et al. GET3D: A generative model of high quality 3D textured shapes learned from images. In Advances in Neural Information Processing Systems, 2022. URL https://arxiv.org/abs/2209. 11163.

Yu Guo, Xin Li, Zhaowen Zhu, et al. MaterialGAN: Reflectance capture using a generative network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. URL https://arxiv.org/abs 2011.12660.

Jun Hong, Bokui Mei, Jianyuan Zeng, et al. LRM: Large reconstruction model for single image to 3D. International Conference on Learning Representations, 2024. URL https://arxiv.org/abs/2311.04400.

Yuanming Hu, Luke Anderson, Tzu-Mao Li, et al. Difftaichi: Differentiable programming for physical simulation. In International Conference on Learning Representations, 2020. URL https://arxiv.org/abs/1910.00935.

Wenlong Huang, Pieter Abbeel, Deepak Pathak, et al. Inner monologue: Embodied reasoning through planning with language models. arXiv preprint arXiv:2207.05608, 2023a. URL https://arxiv.org/abs/2207.05608.

Wenlong Huang, F. Xia, T. Xiao, et al. VoxPoser: Composable 3D value maps for robotic manipulation with language models. arXiv preprint arXiv:2307.05973, 2023b. URL https://arxiv.org/abs/2307.05973.

Zheng Huang, Xianming Wang, et al. Text2Tex: Text-driven texture synthesis via diffusion models. arXiv preprint arXiv:2303.11396, 2023c. URL https://arxiv.org/abs/2303.11396.

Ben T. Jones, Zihan Zhang, Felix Hähnlein, Wojciech Matusik, Maaz Ahmad, Vladimir Kim, and Adriana Schulz. A solver-aided hierarchical language for LLM-driven CAD design. Computer Graphics Forum, 44(7):e70250, 2025. doi: 10.1111/cgf.70250. URL https://doi.org/10.1111/cgf.70250.

R. Kenny Jones, Theresa Barton, Xianghao Xu, Kai Wang, Ellen Jiang, Paul Guerrero, Niloy J. Mitra, and Daniel Ritchie. ShapeAssembly: Learning to generate programs for 3D shape structure synthesis. ACM Transactions on Graphics, 39 (6):234:1–234:20, 2020. doi: 10.1145/3414685.3417812. URL https://doi.org/10.1145/3414685.3417812.

Heewoo Jun and Alex Nichol. Shap-E: Generating conditional 3D implicit functions. arXiv preprint arXiv:2305.02463, 2023. URL https://arxiv.org/abs/2305.02463.

Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3d gaussian splatting for real-time radiance field rendering. In ACM Transactions on Graphics, volume 42, 2023. URL https://repo-sam.inria. fr/fungraph/3d-gaussian-splatting/.

Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. arXiv preprint arXiv:2304.02643, 2023. URL https://arxiv.org/abs/2304.02643.

Samuli Laine, Janne Hellsten, Tero Karras, Yeongwoo Seol, Jaakko Lehtinen, and Timo Aila. Modular primitives for high-performance differentiable rendering. In ACM Transactions on Graphics, volume 39, 2020. URL https: //research.nvidia.com/labs/rtr/nvdiffrast/.

Long Le, Jason Xie, William Liang, Hung-Ju Wang, Yue Yang, Yecheng Jason Ma, Kyle Vedder, Arjun Krishna, Dinesh Jayaraman, and Eric Eaton. Articulate-Anything: Automatic modeling of articulated objects via a vision-language foundation model. In International Conference on Learning Representations, 2025. URL https://arxiv.org/ abs/2410.13882.

Chengshu Li, Jack Liang, Wenlong Shi, et al. Code as policies: Language model programs for embodied control. arXiv preprint arXiv:2209.07753, 2023a. URL https://arxiv.org/abs/2209.07753.

Haitian Li, Haozhe Xie, Junxiang Xu, Beichen Wen, Fangzhou Hong, and Ziwei Liu. MonoArt: Progressive structural reasoning for monocular articulated 3D reconstruction. arXiv preprint arXiv:2603.19231, 2026a. URL https: //arxiv.org/abs/2603.19231.

Ruining Li, Yuxin Yao, Chuanxia Zheng, Christian Rupprecht, Joan Lasenby, Shangzhe Wu, and Andrea Vedaldi. Particulate: Feed-forward 3D object articulation. arXiv preprint arXiv:2512.11798, 2025. URL https://arxiv. org/abs/2512.11798.

Ruining Li, Yuxin Yao, Matt Zhou, Chuanxia Zheng, Christian Rupprecht, Joan Lasenby, Shangzhe Wu, and Andrea Vedaldi. Instruct-Particulate: Scaling feed-forward 3D object articulation with kinematic control. arXiv preprint arXiv:2606.14699, 2026b. URL https://arxiv.org/abs/2606.14699.

Zhaoshuo Li, Thomas Müller, Alex Evans, et al. Neuralangelo: High-fidelity neural surface reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023b. URL https: //arxiv.org/abs/2306.03092.

Chen-Hsuan Lin, Jun Gao, Luming Tang, Towaki Takikawa, Xiaohui Zeng, Xun Huang, Karsten Kreis, Sanja Fidler, Ming-Yu Liu, and Tsung-Yi Lin. Magic3D: High-resolution text-to-3D content creation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2023. URL https://arxiv.org/abs/2211. 10440.

Jiayi Liu, Denys Iliash, Angel X. Chang, Manolis Savva, and Ali Mahdavi-Amiri. SINGAPO: Single image controlled generation of articulated parts in objects. In International Conference on Learning Representations, 2025. URL https://arxiv.org/abs/2410.16499.

Qingming Liu, Xinyue Yao, Shuyuan Zhang, Yueci Deng, Guiliang Liu, Zhen Liu, and Kui Jia. PAct: Part-decomposed single-view articulated object generation. arXiv preprint arXiv:2602.14965, 2026. URL https://arxiv.org/abs/ 2602.14965.

Ruoshi Liu, Rundi Wu, Basile Van Hoorick, Pavel Tokmakov, Sergey Zakharov, and Carl Vondrick. Zero-1-to-3: Zero-shot one image to 3D object. arXiv preprint arXiv:2303.11328, 2023a. URL https://arxiv.org/abs/2303. 11328.

Yuan Liu, Cheng Lin, Zhong Zeng, Xiaoxiao Long, et al. SyncDreamer: Generating multiview-consistent images from a single-view image. arXiv preprint arXiv:2309.03453, 2023b. URL https://arxiv.org/abs/2309.03453.

Xiaoxiao Long, Yuan-Chen Guo, Cheng Lin, Yuan Liu, Zhiyang Dou, Lingjie Liu, Yuexin Ma, Song-Hai Zhang, Marc Habermann, Christian Theobalt, and Wenping Wang. Wonder3D: Single image to 3D using cross-domain diffusion. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024. URL https://arxiv.org/abs/2310.15008.

Viktor Makoviychuk, Lukasz Wawrzyniak, Yunrong Guo, et al. Isaac Gym: High performance GPU-based physics simulation for robot learning. arXiv preprint arXiv:2108.10470, 2021. URL https://arxiv.org/abs/2108. 10470.

Ben Mildenhall, Pratul P. Srinivasan, Matthew Tancik, Jonathan T. Barron, Ravi Ramamoorthi, and Ren Ng. NeRF: Representing scenes as neural radiance fields for view synthesis. In European Conference on Computer Vision, 2020. URL https://arxiv.org/abs/2003.08934.

Kaichun Mo, Shilin Zhu, Angel X. Chang, Li Yi, Subham Tripathi, and Leonidas J. Guibas. PartNet: A large-scale benchmark for fine-grained and hierarchical part-level 3D object understanding. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019. URL https://arxiv.org/abs/1812.02713.

Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics, 41(4), 2022. URL https://arxiv.org/abs/ 2201.05989.

Alex Nichol, Heewoo Jun, Prafulla Dhariwal, Pamela Mishkin, and Mark Chen. Point-E: A system for generating 3D point clouds from complex prompts. arXiv preprint arXiv:2212.08751, 2022. URL https://arxiv.org/abs/ 2212.08751.

Maxime Oquab, Timothée Darcet, Théo Moutakanni, et al. Dinov2: Learning robust visual features without supervision. In Transactions on Machine Learning Research, 2024. URL https://arxiv.org/abs/2304.07193.

Ben Poole, Ajay Jain, Jonathan T. Barron, and Ben Mildenhall. DreamFusion: Text-to-3D using 2D diffusion. International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2209.14988.

Guocheng Qian, Jinjing Mai, Hasan Hammoud, Antonio Alliegro, Cordelia Schmid, and Bernard Ghanem. Magic123: One image to high-quality 3D object generation using both 2D and 3D diffusion priors. arXiv preprint arXiv:2306.17843, 2023. URL https://arxiv.org/abs/2306.17843.

Xiaowen Qiu, Jincheng Yang, Yian Wang, Zhehuan Chen, Yufei Wang, Tsun-Hsuan Wang, Zhou Xian, and Chuang Gan. Articulate AnyMesh: Open-vocabulary 3D articulated objects modeling. In Proceedings ofThe 9th Conference on Robot Learning, volume 305 of Proceedings of Machine Learning Research, pp. 4585–4606, 2025. URL https://proceedings.mlr.press/v305/qiu25b.html.

Alec Radford, Jong Wook Kim, Chris Hallacy, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, 2021. URL https://arxiv.org/abs/2103. 00020.

Yichun Shi, Peng Wang, Jianglong Ye, Mai Long, Kejie Li, and Xiao Yang. MVDream: Multi-view diffusion for 3D generation. arXiv preprint arXiv:2308.16512, 2023. URL https://arxiv.org/abs/2308.16512.

Ishika Singh, Valts Blukis, Arsalan Mousavian, et al. ProgPrompt: Generating situated robot task plans with large language models. arXiv preprint arXiv:2209.11302, 2023. URL https://arxiv.org/abs/2209.11302.

Jiaxiang Tang et al. LGM: Large multi-view gaussian model for high-resolution 3D content creation. arXiv preprint arXiv:2402.05054, 2024. URL https://arxiv.org/abs/2402.05054.

Junshu Tang, Meng Ren, Hang Zhou, Ziwei Liu, and Gang Zeng. Make-It-3D: High-fidelity 3D creation from a single image with diffusion prior. arXiv preprint arXiv:2303.14184, 2023. URL https://arxiv.org/abs/2303.14184.

Sergey Tochil et al. TripoSR: Fast 3D object reconstruction from a single image. arXiv preprint arXiv:2403.02151, 2024. URL https://arxiv.org/abs/2403.02151.

Emanuel Todorov, Tom Erez, and Yuval Tassa. MuJoCo: A physics engine for model-based control. In 2012 IEEE/RSJ International Conference on Intelligent Robots and Systems, pp. 5026–5033. IEEE, 2012. doi: 10.1109/IROS.2012. 6386109. URL https://doi.org/10.1109/IROS.2012.6386109.

Karthik Valmeekam, Matthew Marquez, and Subbarao Kambhampati. PlanBench: An extensible benchmark for evaluating large language models on planning and reasoning about change. arXiv preprint arXiv:2206.10498, 2023. URL https://arxiv.org/abs/2206.10498.

Vikram Voleti, Chun-Han Yao, Mark Boss, Adam Letts, Albert Pumarola, Varun Jampani, and Yan Yan. SV3D: Novel multi-view synthesis and 3D generation from a single image using latent video diffusion. arXiv preprint arXiv:2403.12008, 2024. URL https://arxiv.org/abs/2403.12008.

Rui-Huan Wang, Si-Tong Wei, Jia-Qi He, Heng-Yi Wei, Baoquan Chen, and Peng-Shuai Wang. aDSL: Agentic 3D creation via joint agent-program design. arXiv preprint arXiv:2608.17975, 2026. URL https://arxiv.org/abs/ 2608.17975.

Xiaogang Wang, Bin Zhou, Yahao Shi, Xiaowu Chen, Qinping Zhao, and Kai Xu. Shape2Motion: Joint analysis of motion parts and attributes from 3D shapes. arXiv preprint arXiv:1903.03911, 2019. URL https://arxiv.org/ abs/1903.03911.

Guanjun Wu, Jiemin Fang, Chen Yang, Sikuang Li, Taoran Yi, Jia Lu, Zanwei Zhou, Jiazhong Cen, Lingxi Xie, Xiaopeng Zhang, et al. Unilat3d: Geometry-appearance unified latents for single-stage 3d generation. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4366–4378, 2026.

Fei Xia, Yixin Zhu, Konstantinos Bousmalis, et al. PartNet-Mobility: A benchmark for learning articulated object models from point clouds. arXiv preprint arXiv:2004.04880, 2020. URL https://arxiv.org/abs/2004.04880.

Fu Xiang, Frederik Ebert, Corey Cheng, et al. SAPIEN: A simulated part-based interactive environment. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020. URL https://arxiv.org/abs/ 2003.08515.

Jianfeng Xiang, Xiaoxue Chen, Sicheng Xu, Ruicheng Wang, Zelong Lv, Yu Deng, Hongyuan Zhu, Yue Dong, Hao Zhao, Nicholas Jing Yuan, and Jiaolong Yang. Native and compact structured latents for 3D generation. arXiv preprint arXiv:2512.14692, 2025a. URL https://arxiv.org/abs/2512.14692.

Jianfeng Xiang, Zelong Lv, Sicheng Xu, Yu Deng, Ruicheng Wang, Bowen Zhang, Dong Chen, Xin Tong, and Jiaolong Yang. Structured 3D latents for scalable and versatile 3D generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 21469–21480, 2025b. URL https://openaccess.thecvf.com/content/CVPR2025/html/Xiang\_Structured\_3D\_Latents\_for\_ Scalable\_and\_Versatile\_3D\_Generation\_CVPR\_2025\_paper.html.

Jiale Xu et al. InstantMesh: Efficient 3D mesh generation from a single image. arXiv preprint arXiv:2404.07191, 2024. URL https://arxiv.org/abs/2404.07191.

Chen Yang, Jiemin Fang, Fanpeng Meng, Zihao Huang, Jichen Pang, Guanjun Wu, Zhikuan Bao, and Qi Tian. SUPRA-Tex: 3d generation with super-aligned texture. arXiv preprint arXiv:2610.22352, 2026. URL https: //arxiv.org/abs/2610.22352.

Qiaojun Yu, Ce Hao, Xibin Yuan, et al. GAPartNet: Cross-category generalizable object perception and manipulation using 3D part affordance. arXiv preprint arXiv:2211.05272, 2023. URL https://arxiv.org/abs/2211.05272.

Matt Zhou, Ruining Li, Xiaoyang Lyu, Zhaomou Song, Zhening Huang, Chuanxia Zheng, Christian Rupprecht, Andrea Vedaldi, and Shangzhe Wu. Articraft: An agentic system for scalable articulated 3D asset generation. arXiv preprint arXiv:2605.15187, 2026. URL https://arxiv.org/abs/2605.15187.

## Supplementary Material

This supplement provides implementation details, evaluation protocols, and an extended qualitative comparison accompanying the main paper.

## A Method and Implementation Details

## A.1 Interaction Representation

The representation in Section 3.2 separates interaction intent, mechanism design, and concrete assets. The semantic level S identifies the target region, action, expected response, and observation frame. The mechanism level K specifies bodies or compliant domains, connections, degrees of freedom, supports, and parameter assumptions. The realization level X identifies the meshes, materials, scene entities, controls, and exported bindings. The relations Γ connect these levels; ρ records the origin and modification of source surfaces; and E associates observations with the candidate that produced them. These correspondences link interaction requirements to mechanisms, realized entities, and surface bindings for construction and feedback-guided revision, as defined in Section 3.2. The following subsections detail the associated construction and validation procedures.

For example, a lid-opening requirement refers to the lid region and measures its motion relative to the base. Its mechanism identifies the hinge axis and permitted range, while its realization identifies the moving body and the visible lid faces assigned to it. This correspondence allows the hinge configuration and visible-surface assignment to be inspected separately; it localizes the relevant entities without by itself establishing the cause of a failure. For a padded region, the corresponding records identify the load, support, physical discretization, and visible skin. These examples explain the representation without assuming that hidden mechanisms or material parameters are uniquely recoverable from the input image.

## A.2 Geometry, Materials, and Surface Binding

The initial TRELLIS.2 reconstruction supplies the detailed visible geometry and materials. Geometric refinement uses rendered views and measurements to identify actionable regions: mesh connectivity alone does not determine which parts should move, and a texture boundary need not be a mechanical boundary. Bounded edits preserve useful source faces, UV coordinates, and material assignments. When topology changes, source correspondence must be updated rather than inherited from obsolete face indices.

For rigid articulation, visible vertices follow the transform of their assigned body. Correct body ownership, local transforms, parent–child relationships, joint frames, and limits are all needed to produce the intended visible motion. Collision geometry can differ from the detailed render mesh, but motion of a collision proxy alone does not verify motion of the visible part.

For compliant regions, the physical discretization and visible skin may also differ. The displacement binding in Eq. 4 transfers physical-node displacements to visible vertices in a common frame. Its node identities, dimensions, weights, and rest geometry are checked against the exported asset. Supports and loading conditions are specified separately from the skin binding. A local deformation observation removes common rigid motion before assessing shape change; otherwise, movement of the entire object could be mistaken for local compliance. The ear-pad example uses an approximate compliance model rather than a material calibration recovered from the image.

## A.3 Feedback and Candidate Records

Rendered observations assess appearance, part separation, and visible response. Execution observations assess the requested action, achieved range, contacts, supports, and applicable deformation or recovery. Each record identifies the requirement, candidate, target region and entities, action, initial state, runtime and assessment configurations, and assessment outcome. The four outcomes are unassessed, pass, fail, and inconclusive. An execution error supplies no measured trajectory, and a changed build requires renewed dynamic evidence.

Feedback associates a discrepancy with the relevant mechanism, realized entity, or binding. Local repairs can address motion range, tracking, surface assignment, or assembly; structural replanning addresses topology, ownership, joint-axis, or material contradictions. Previous candidates remain available when a revision fails. The budgets and selection procedure below bound this loop.

Table 6. Documented implementation configuration and construction budgets. Host validation attempts and simulations called inside construction tools are counted separately.
<table><tr><td>Component</td><td>Configuration</td></tr><tr><td>3D generator</td><td>TRELLIS.2-4B; 1024_cascade route</td></tr><tr><td>Sampling and texture</td><td>24 sampling steps; 2048-pixel texture size</td></tr><tr><td>Generator seed</td><td>42</td></tr><tr><td>Reasoning model</td><td>gpt-6-astra; high reasoning effort</td></tr><tr><td>Geometry / simulation</td><td>Blender / MuJoCo</td></tr><tr><td>Builder attempts</td><td>3 total: initial construction and at most 2 repairs</td></tr><tr><td>Host physical validations</td><td>At most 2 attempts</td></tr><tr><td>Structural replanning</td><td>At most 1 replan</td></tr><tr><td>Planner / Reviewer responses Returned role messages</td><td>At most 2 / 3 responses</td></tr><tr><td></td><td>At most 28</td></tr><tr><td>Joint-range reductions</td><td>At most 2 per joint</td></tr></table>

## A.4 Runtime Configuration and Budgets

Table 6 summarizes the documented v0.2.0 runtime configuration. Blender handles geometry and material editing; MuJoCo executes supported physical models. The reasoning model is accessed through the local Codex interface. The configured generator seed identifies the initialization setting, without implying deterministic agent execution.

Invalid returned messages consume their role-response allowance. Host validation attempts are reserved before execution, so an interrupted simulation also consumes an attempt. Candidate history and cumulative counters persist across continuations; selecting an earlier candidate does not reset them. Simulator calls made inside construction tools are not covered by the host-attempt limit. These counts therefore describe distinct resources rather than an aggregate computation budget. The initial construction and at most two repairs constitute the Builder budget in Section 4.1; structural replanning does not reset this budget. As in Section 3.4, a modified build requires new dynamic evidence, and a preview does not turn an unverified simulation status into a pass.

## A.5 Motion-Range Adaptation and Physical Checks

For a hinge or slide with original interval $[ \ell _ { 0 } , h _ { 0 } ]$ and neutral coordinate $q _ { 0 }$ , the documented range helper tests

$$
\ell _ { n } = q _ { 0 } + ( \ell _ { 0 } - q _ { 0 } ) ( 2 / 3 ) ^ { n } , \qquad h _ { n } = q _ { 0 } + ( h _ { 0 } - q _ { 0 } ) ( 2 / 3 ) ^ { n } , \quad n \in \{ 0 , 1 , 2 \} .\tag{6}
$$

The original interval remains the reference across repairs. The retained span must be at least $0 . 2 ( h _ { 0 } - \ell _ { 0 } )$ and, for a hinge, at least 10<sup>◦</sup>. A reduced interval is considered only after a completed failed sweep with a valid rest pose. The checks include rest and both motion directions where applicable. A colliding rest pose, missing clearance information, incomplete sweep, or execution error terminates this search. The achieved range is retained explicitly; reducing travel does not satisfy an unchanged full-closure requirement.

Let D be the positive asset diagonal in meters. The documented acceptance profile uses a contact-proxy penetration tolerance of 0.005D, an angular endpoint tolerance of $5 ^ { \circ }$ , and a translational endpoint tolerance of 0.005D. Additional checks cover finite states, solver warnings, joint limits, measured interaction, and applicable compliant-region requirements. Geometric sweep checks apply to sampled configurations and do not replace dynamic execution.

For a declared compliant region, a least-squares rigid alignment removes common motion from the simulation nodes. Let $d _ { \mathrm { m a x } }$ be the maximum residual over recorded samples and phase endpoints. The response criterion requires $d _ { \operatorname* { m a x } } ~ \ge ~ 0 . 0 0 1 D$ together with a pairwise-distance-change check. When recovery is required, the residual after unloading must not exceed $\operatorname* { m a x } ( 0 . 0 0 1 D , 0 . 3 5 d _ { \operatorname* { m a x } } )$ . These criteria are deformation heuristics on simulation nodes; exported-surface response is assessed separately through the binding and visible observations.

## A.6 Output Selection

Selection excludes unloadable, empty, nonfinite, externally dependent assets and detected source-identity conflicts. In autonomous runs, eligible candidates are ranked by appearance evidence, native retention, semantic evidence, absence of unsupported additions, display checks, declared bodies, supported physics acceptance, and recency, in that order. Native retention includes source identity, face accounting, geometry, and UV/material checks. As specified in Section 4.5, Full pipeline and w/o surface preservation use the same selection rule. In the latter, source-geometry, UV, and material preservation constraints are removed from construction, while the shared ranking criteria and physical-to-visible bindings remain. This intervention therefore tests the construction constraints under a fixed output-selection policy. Supported physics acceptance requires a passing report, at least three trajectory states, and matching asset and trajectory identities. A preview can be returned without a valid dynamic trajectory, with that status kept distinct. If no refined candidate is eligible, the native reconstruction can serve as a static fallback. Such a fallback is not evidence that the requested interaction or dynamic validation succeeded; failed and inconclusive attempts remain in the outcome accounting.

## B Evaluation and Comparison Details

## B.1 Evaluation Scope and Aggregation

The in-the-wild collection contains more than 200 photographs; the fixed evaluation subset contains 50 objects and 110 predefined semantic actions. Following Section 4.2, ACR, MMR, and RAR are macro-averaged over objects. For normalized object-level scores s (m) , metric m is reported as $\begin{array} { r } { 1 0 0 N ^ { - 1 } \sum _ { i = 1 } ^ { N } s _ { i } ^ { ( m ) } } \end{array}$ , with $N = 5 0$ . Thus, each object contributes equally, regardless of its number of rendered frames.

ACR measures correct semantic action completion. Strict MMR counts missing, incomplete, incorrectly related, structurally broken, or spuriously coupled actions. RAR scores the maximum valid visible range in {0, 0.25, 0.5, 0.75, 1} after the ACR semantic gate. FVSP is the percentage of objects that complete every expected action without video-visible instability. It is a joint functional-and-visible-stability criterion based on rendered videos, not simulator logs.

Support marks indicate declared or evaluated capabilities, not successful outputs. Static assets remain evaluable for input consistency; missing or invalid interaction videos count as failures within the fixed action denominator. Failed and inconclusive attempts remain in the outcome accounting, with unsupported output types distinguished in the records. Development examples remain separate from the benchmark comparison.

## B.2 Inputs, Baselines, and Scoring

Section 4.1 specifies the common comparison protocol. Image-conditioned methods receive the same photograph; mesh-conditioned methods require an identified reconstruction and any auxiliary conditioning. Format and coordinate conversions are distinguished from geometry repair or inference of missing physical parameters, which would change the evaluated method. PhysXGen denotes the framework introduced in PhysX-3D (Cao et al., 2025a). aDSL reports imageto-shape generation and articulated creation as a downstream application, motivating its inclusion in the single-image comparison.

Reference-based comparisons use compatible alignment, pose, sampling, distance normalization, and scoring settings. Shape normalization must preserve the separately assessed absolute-scale error. Table 2 follows the PhysXVerse/PhysX-Mobility split reported by PhysX-Omni; its kinematic score is not a dynamics success rate. Missing scores, marked by a dash or NR, are unreported entries rather than measured zeros.

PhysX-Bench uses fixed evidence generation, evaluator configuration, and prompts across methods. Joint playback supplies articulation evidence, while material response and static appearance are separate observations; playback does not establish dynamic contact stability. The selected states in Figure 4 likewise demonstrate visible articulation and surface appearance. Table 7 summarizes the evaluation settings.

## B.3 Ablation Configuration

The three variants in Section 4.5 share inputs, cached reconstructions, initial priors, model and decoding settings, construction tools, and final scoring. Full pipeline and w/o surface preservation share the two-repair limit and candidateselection rule. One-shot returns the initial candidate and receives final scoring without feeding it back for repair. All attempted inputs, including failed and inconclusive runs, remain in the outcome accounting.

The source-preservation ablation removes construction-time geometry, UV, and material constraints while retaining the initial reconstruction and physical-to-visible bindings. It keeps output selection fixed, including the native-retention ranking criterion. The One-shot comparison measures refinement as a whole, including additional repair and candidateselection opportunities; it does not isolate an individual feedback signal.

## C Additional Qualitative Comparisons

Figure 5 extends Figure 3 with additional cases and methods, showing geometry, materials, and texture continuity alongside input-image agreement in proportions, parts, colors, and patterns. No additional aggregate score is introduced.

## PhysAg3D

Table 7. Evidence and interpretation for the evaluations reported in the main paper. The rows describe evaluation settings, not additional experiments.
<table><tr><td>Evaluation</td><td>Evidence and interpretation</td></tr><tr><td>In-the-wild interaction</td><td>ACR, MMR, and RAR: macro averages over 50 objects, computed from 110 predefined actions. FVSP: object-level pass rate requiring all actions to complete without video-visible instability.</td></tr><tr><td>PhysXVerse / PhysX-Mobility</td><td>Reference assets support geometry, appearance, scale, and attribute comparisons. CD uses 10−3 units; F-score uses 10−2units at threshold 0.05, as in Table 2.</td></tr><tr><td>PhysX-Bench</td><td>CLIP, RQS, and MCS address alignment, visual quality, and multi-view consistency. DQS addresses scale; APS, KPS, MPS, and DCS address affordance, kinematics, materials, and description.</td></tr><tr><td>Component ablation</td><td>The same PhysX-Bench setting compares Full pipeline, One-shot, and w/o surface preservation using CLIP, RQS, KPS, and MPS.</td></tr><tr><td>Extended visualization</td><td>Additional real-world cases and methods support inspection of photorealistic appearance and correspondence to the input image.</td></tr></table>

![](images/8c987c271dc93c8bb9c34dfcf7043c67bb4d1a111fcc06a001b9c30a3ed9e220.jpg)

Figure 5. Extended qualitative comparison on real-world images. Complementing Figure 3, the expanded comparison focuses on photorealistic appearance and alignment with the input photograph across additional cases and methods.