# Open-Form Geometric Style Transfer

Adar Elad<sup>1∗</sup>

Gal Metzer<sup>1</sup>

Omer Dahari<sup>1</sup>

Min Lu<sup>2</sup>

Daniel Cohen-Or<sup>1</sup>

<sup>1</sup>Tel Aviv University

<sup>2</sup>Shenzhen University

{elad.research, gal.metzer, omer11a, lumin.viz, cohenor}@gmail.com

![](images/7a9e4e049c09cffc061ac9d830b319f7be6d79b1365bb507b990d1ac0d81f191.jpg)  
Source Style  
Open-Form Synthesis

Figure 1. Open-Form Geometric Style Transfer. Given a small set of examples sharing a geometric style (left), we learn their common geometric design language and apply it to unseen structural sketches. The resulting forms (right) span challenging and unrelated shape families, illustrating open-form transfer beyond the family in which the style was originally observed.

## Abstract

A geometric style learnedfrom one family of shapes is typically applied to new shapesfrom the samefamily. We consider a more challenging setting: learning the geometric design language of a stylized Latin alphabet and applying it to new forms, from Greek glyphs to entirely different structures such as human poses. We introduce Open-Form Geometric Style Transfer, where the forms to which a learned geometric style can be applied are not restricted to the shape family in which that style was observed. This setting is deceptively challenging: the difficulty is not merely learning a style, but learning the style without learning the shape family in which it was observed. We represent target forms by structural sketches and disentangle structural adherence from style realization using a Geometry LoRA and a stylespecific LoRA. Critically, open-form transfer requires the

Geometry LoRA to be universal. To achieve this, we devise a geometry-enrichment strategy that constructs a large and purposefully diverse collection of synthetic structural sketches spanning a broad range of unfamiliar geometries. Our experiments and ablations demonstrate the importance of both disentanglement and this universal geometry prior, enabling geometric styles learnedfrom glyphs, pictograms, and other shape families to be transferred to substantially different, unseen forms.

## 1. Introduction

Style transfer is a long-standing and intriguing problem in visual computing: given examples that contain both content and style, the goal is to capture the essence of the style, separate it from the particular content on which it is observed, and apply it to new content. The separation of style from content has been studied for decades [7, 8], and has more recently become central to neural style transfer and generative modeling [13, 15, 27]. At the heart of the problem lies the ability to distinguish what is represented from how it is expressed.

![](images/0fa17243342d0188cc9f8d3ab6da46b8a6e4a87bb832a8148e4d330c2fbb17ee.jpg)  
Figure 2. Open-Form Geometric Style Transfer. Given a set of glyph examples sharing a geometric style (left), we learn to apply that style to new structural sketches (right). The target forms are unseen during style learning and may belong to entirely different shape families.

In this work, we focus on geometric style, where style governs how an underlying structure is geometrically realized. A canonical example is a font: different glyphs have different structures, yet their geometric realizations follow a coherent design language that makes them immediately recognizable as belonging to the same font. More generally, such geometric styles cannot be inferred from a single example; it is revealed through a collection of different shapes that share the same design language. Early work has considered related notions of separating style from content and transferring geometric style across shapes [9, 25].

Existing approaches [22, 26, 29] typically learn and apply such styles within a shared domain: a style observed on letters is transferred to other letters, and a style observed on human figures is applied to other human figures. We remove this restriction and consider transferring a geometric style to an open space of forms. As illustrated in Figure 1, a style learned from a Latin alphabet may be applied to Greek glyphs, or more radically, to human poses; conversely, a geometric style learned from pictograms may be applied to glyphs. We call this new setting Open-Form Geometric Style Transfer, where the forms to which a learned geometric style can be applied are not restricted to the shape family in which that style was observed. We focus on forms whose underlying structure can be represented by a sparse structural sketch, which serves as the geometry to be styled.

The open-form setting, illustrated in Figure 2, is deceptively challenging. The difficulty is not simply learning a style, but learning the style without learning the shape family in which that style was observed. Existing font-generation methods can transfer styles to unseen glyphs [18, 26, 29, 32], and some explicitly separate character content from style. However, their structural component is learned and evaluated largely within the glyph domain, often within a single writing system. This distinction becomes critical when the target form leaves the source domain. A straightforward LoRA trained on a limited styled shape family learns an entangled style with the geometries observed at training (Sec. 4.2), causing it to drift toward familiar source-domain shapes when presented with substantially different target geometries.

Our approach addresses this entanglement by separating structural adherence from style realization. A Geometry LoRA is responsible for faithfully following the target structural sketch serving as a geometric-adherence prior, while a Style LoRA learns how that geometry should be realized in the desired style. Accordingly, the Geometry LoRA must be universal: generalize across the vast universe of styles, responding to structure rather than shape identity or residual stylistic cues. Acquiring such a universal geometry prior is itself non-trivial: a corpus of ordinary glyphs provides a limited and familiar geometry vocabulary, allowing the model to recognize known shapes rather than truly follow the input structure. We therefore introduce a geometry-enrichment strategy (GEE Sec. 3.2) that systematically leverages SVG font data to construct a large collection of unfamiliar synthetic geometries, untangled from their source style. This strategy proves vital for successfully learning a universal geometry LoRA.

Our contributions are threefold:

• We introduce Open-Form Geometric Stylization, where a geometric style learned from one shape domain is applied to universally arbitrary forms.

• We show that open-form stylization requires separating

structural adherence from style realization.

• Finally, we introduce a geometry-enrichment strategy for constructing the broad and diverse structural training distribution needed to acquire this universal prior.

We evaluate these ideas through controlled ablations and increasingly challenging transfers, from held-out glyphs and cross-alphabet transfer to unrelated forms such as human poses, demonstrating that a geometric style can be extended well beyond the family of shapes on which it was observed.

## 2. Background and Related Work

Our work builds on three lines of research: the separation of style and content, font and glyph style transfer, and structurally conditioned generation. Below we review each line, clarify how our notion of geometric style relates to them, and how they relate to our approach.

## 2.1. Style and Content

Our work is closely connected to the seminal work of Tenenbaum and Freeman [25] on separating style and content. Using bilinear models, they showed that a common style can be inferred from multiple observations and applied to new content. Particularly relevant to our setting is their treatment of fonts, where style is shared across different letters and can be used to synthesize previously unseen characters. This provides an early and elegant formulation of style as a factor that can be learned across a family of shapes and separated from their individual geometry. Our work builds on this perspective, but considers an open-form setting in which the new content need not belong to the same family: a geometric style learned from one alphabet may be applied to another alphabet, or even to a different class of structures such as human poses.

More broadly, style transfer has a long history in computer vision and graphics, with much of the literature focusing on style as visual appearance. Early example-based methods [7, 8] synthesized and transferred texture by sampling local neighborhoods or patches from example images. [13] later introduced neural style transfer, demonstrating that content and artistic style can be represented separately within a deep network and recombined to render new content in the appearance of a reference image. Recent diffusion-based methods such as StyleAligned [15], B-LoRA [10], InstantStyle [28], and ZipLoRA [24] substantially extend the quality and flexibility of referencebased style transfer. Across these works, style is predominantly expressed through visual appearance, including texture, color, material, brushwork, and shading.

Our notion of geometric style is different. Here, style determines how an underlying structure is geometrically realized: how curves are formed, junctions are connected, strokes vary in thickness, gaps are introduced, and structural elements terminate or interact. Such a style is naturally revealed by a set of related shapes rather than by a single reference image. Glyphs provide a particularly clean instance of this setting: individual characters differ in structure, while together they reveal a consistent geometric design language. Related notions of transferring geometric style across shapes were also explored in early work by [9]. More recently, geometric stylization has also been explored for 3D shapes [5].

Closely related, Abstraction in Style (AiS) [19] also separates geometry from appearance using two specialized LoRA adapters. However, AiS learns style-specific geometric abstraction, whereas our Geometry LoRA is universal, enabling a learned geometric style to generalize to unseen structural sketches across shape families.

## 2.2. Font and Glyph Style Transfer

Modern font-generation methods provide the closest setting to our work, as they explicitly transfer a common style across glyphs with different structures. FontDiffuser [29] formulates one-shot diffusion-based font generation, using multi-scale content aggregation to preserve character structure and contrastive learning to capture style. It generalizes to unseen characters and also demonstrates cross-lingual generation. MX-Font [22] addresses few-shot font generation using multiple localized experts that disentangle content and style while capturing different local style characteristics. It also demonstrates cross-lingual generalization to unseen writing systems. VecFusion [26] approaches fon generation in the vector domain, explicitly modeling glyph topology, paths, and Bézier control points, and supports applications including glyph completion and few-shot style transfer. Skeleton-Guided Diffusion [32] is particularly related to our structural representation, using character skeletons to guide diffusion and preserve structure during font generation. StrokeStyles [1] decomposes font glyphs into meaningful stroke structures and uses this representation for consistent geometric stylization across characters.

FLUX-Font [18] is particularly close to our work architecturally. It adapts a pre-trained diffusion transformer using separate LoRA modules [17] for glyph content and style, explicitly separating the two factors rather than learning them through a single adapter. Our method shares this general motivation, but separates structural adherence from style realization hierarchically: a universal Geometry LoRA is first trained to follow arbitrary structural sketches, and each Style LoRA is subsequently learned with this geometry prior active and fixed.

Despite their differences, these methods share an important restriction: their structural representations and learned priors are designed for the glyph domain. Their goal is to generalize from observed characters to other characters, possibly across writing systems, but the target remains a glyph. Open-form transfer removes this restriction. The target structural sketch may describe a character, a pictogram, a human pose, or another unrelated form. Consequently, separating content from style within the font domain is not sufficient; the geometry-following component must itself generalize beyond the shape family in which the style was observed.

![](images/7dca580eb0e9f430967feb57d831ca97549b362b73049f66bf62a92b0734e89d.jpg)  
Figure 3. Method overview. (i) GEE expands a small vocabulary of familiar glyphs into a large corpus of unfamiliar geometries paired with style-suppressed structural sketches. (ii) The Universal Geometry LoRA (UGL) is trained once on this corpus and then frozen, providing a general geometry-following prior. (iii) Each Style LoRA is trained on top of the frozen UGL using only a few style examples, with geometry reinforcement preserving its open-form capability. (iv) At inference, UGL and the Style LoRA are applied jointly over the open-form inputs

## 2.3. Structural Conditioning

Spatially conditioned diffusion models provide a complementary line of related work. ControlNet [31] and T2I-Adapter [20] condition pre-trained text-to-image models on spatial signals such as edges, depth, segmentation, poses, and sketches, while Sketch-Guided Diffusion [27] focuses specifically on generation controlled by free-form sketches. These methods demonstrate that strong spatial adherence can be learned over broad image distributions.

Our problem, however, is not spatial conditioning alone. The structural sketch specifies a form whose geometric realization must change according to a style learned from a separate set of examples. The challenge is therefore to preserve the target structure while preventing the learned style from carrying with it the shape distribution on which that style was observed. We use structural conditioning as a means toward this goal, learning a universal geometry prior that remains fixed while different geometric styles are acquired. At inference, we treat structure and style as independent conditions and adopt the dual-conditioning guidance of InstructPix2Pix [3], allowing their strengths to be controlled separately.

## 3. Method

Given a collection of shapes sharing a geometric style, our goal is to learn that style and apply it to a new target form specified by a structural sketch. We use the term sketchglyph for a pair consisting of a structural sketch and its styled realization. The term originates naturally from fonts, where a sketch specifies the structure of a character and the glyph realizes it in a particular font. In this work, we extend this notion to pictograms, symbols, and other structured forms that are not letters. Given a small collection of sketch-glyph pairs sharing a style, we aim to realize arbitrary new structural sketches in that style, as illustrated in Fig. 2.

Figure 3 provides an overview of the method. Our key design is to separate the two responsibilities involved in this task. A Universal Geometry LoRA (UGL) provides a general ability to faithfully follow the input structural sketch, while a Style LoRA learns how the conditioned geometry should be realized in the target style. The two adapters are learned hierarchically rather than independently. UGL is trained once (Fig. 3 ii); each Style LoRA is subsequently trained on top of the frozen UGL (Fig. 3 iii).

For this separation to support open-form transfer, UGL itself must generalize far beyond any particular shape family. We therefore train it on a large and purposefully diverse collection of synthetic geometries constructed by our Geometry Enrichment Engine (GEE) (Fig. 3 i).

At inference (Fig. 3 iv), UGL and the selected Style LoRA are applied jointly. Their contributions can be controlled independently through separate geometry and style guidance scales, allowing us to balance adherence to the target structure against the strength of its stylistic realization. The following sections describe UGL, GEE, and Style LoRA training in detail.

![](images/eff0fcf6f187efacab3d7c0ac3b837b9f997aa1b5b256cd0cb047008e37fc451.jpg)  
Figure 4. GEE overview. (a) Geometry generation recursively merges and removes glyph components in font i, recording a fontindependent recipe. (b) Reapplying the recipe to font j produces the same underlying construction in another style. (c) Skeleton extraction yields a conditioning sketch from j, which is paired with the target glyph from i.

## 3.1. Universal Geometry LoRA (UGL)

The purpose of UGL is to fine-tune a text-to-image diffusion model to follow open-form geometries (i.e. sketches). Starting from a pre-trained text-to-image diffusion transformer (DiT), we fit LoRA parameters on the DiT’s weights using sketch-glyph data pairs constructed by GEE. The rich, diverse synthetic data produced by the GEE allows us to select a high LoRA rank without overfitting, giving the UGL significant learning capacity. After this one-time training, we freeze UGL and share it across all Style LoRAs, providing a common geometry prior beyond each style’s specific training examples.

## 3.2. Geometry Enrichment Engine (GEE)

Fonts are a natural starting point for data augmentation: they are available in a large variety of styles, and more importantly, their SVG native representation enables truly geometric augmentations compared to raster form. However, using real font glyphs directly presents two limitations. First, available fonts cover a relatively narrow set of languages and therefore a limited vocabulary of structures, which may hinder generalization to unseen scripts and, even more, to out-of-domain forms such as symbols or pictograms. Second, the geometry of a glyph is itself tied to the style of its source font, so the conditioning structure may carry unwanted stylistic cues. To address both issues, we introduce GEE: a data-generation procedure that expands the range of training geometries while reducing their dependence on font style. To increase structural diversity, GEE takes advantage of the native SVG parameterization of fonts and constructs novel geometric forms by applying random sequences of geometric edits. We refer to each such sequence as a geometric recipe. A recipe may, for example, repeatedly remove or merge SVG parts, producing coherent but unfamiliar structures (Fig. 4(a)). Recipes also allow the GEE to address the second problem. A recipe applied to one font can be re-applied to matching letters from another font, producing paired data with similar structure but different styles (Fig. 4(b)). During UGL training, we form mixed pairs by taking the conditioning sketch from one font and the target from another font to which the same recipe was applied (Fig. 4(c)). The sketch and target therefore remain aligned in structure while differing in style, encouraging UGL to learn the supplied geometry independently of its stylistic realization.

We next describe the editing primitives that drive the geometry enrichment process, first motivating their design and then outlining each one (full details in Appendix C).

Structural editing primitives. A glyph’s structure is determined by how its strokes relate: which strokes meet and where they branch or terminate. Smooth transformations such as warping, shearing, or scaling can change a glyph’s appearance, but they do not create new structural relations between its strokes. To generate genuinely new structures, we therefore need operations that modify these relations. We propose two elementary edits: component removal and merging. This builds on a broader principle: deleting and recombining existing parts can generate meaningful structural variation [11], and related cut-based augmentations have been shown to improve recognition performance [6, 30]. Repeating these edits produces novel structural relations beyond those found in existing alphabets.

To make both edits transferable across fonts, we edit the glyph’s structure rather than its contour, which is more sensitive to stylistic variation. We therefore represent each glyph as a graph, with strokes as edges and intersections and endpoints as nodes. A recipe is then expressed as a sequence of graph edits, with component removal and merging each applied one to three times. The edit type and its structural parameters are sampled when the recipe is created, specifying which component to remove or where components should connect. Because the edits are defined structurally, the same recipe can be replayed across fonts to produce the same underlying structure in multiple styles. Structural correspondence across styles is further explained in Appendix C.

Graph extraction. We represent each glyph as a graph that captures its dominant stroke structure (Fig. 5). Starting from the glyph (a), we compute its medial axis together with the local inscribed-disk radius, which provides an estimate of stroke thickness (b). We prune short branches caused by serifs and other stylistic details, leaving the main stroke structure (c). Pixel chains are then collapsed into edges, with junctions and stroke endpoints represented as nod es (d). Finally, we classify each edge by its structural role: an edge between two junctions is a bar, while an edge from a junction to an endpoint is a leg (e).

![](images/037d6a2da0fa990aef80b3774c12d46bd1f4ac193bab16662965c2d20066e0ab.jpg)  
Figure 5. Graph extraction. The steps of extracting a structural graph from a glyph, shown for two glyphs. Columns labelled as referenced in Sec. 3.2.

Component removal. Fig. 6 illustrates the process. Our strategy is to perform the edit in the medial-axis representation and then reconstruct the glyph back in the SVG domain. Outline points unrelated to the removed component are preserved exactly. Only the region surrounding the cut is reconstructed, where additional processing is needed to obtain a clean, style-consistent boundary. Using the graph extracted in the previous step, we locate the component to remove from its structure, which lets us find the same component across styles (a). As seen in Fig. 6(b), simply removing the red branch can leave bulges where it connects to the remaining blue branches. We therefore trim these regions and reconnect the remaining blue branches. We then separate the pixels belonging to the removed region from those that should remain (c). This produces a pixel removal mask, which we convert to a smooth vector representation (d).Additional reconstruction steps are described in the appendix. The final removal result is shown in (e).

![](images/b156420fa2f95088ab8db058ab40d4170ec2a3897cab0f86381887bb51847e24.jpg)  
Figure 6. Component removal. The steps of removing one structural component, shown for two glyphs. Columns labelled as referenced in Sec. 3.2.

Merging. Fig. 7 illustrates the process. We begin with two glyphs from the same style: a guest (a) and a host (b), and aim to combine them into a single coherent glyph. As in component removal, we perform the edit in the medialaxis representation and reconstruct the result in the SVG domain. Using the graphs extracted in the previous step, we identify where the guest should connect to the host based on their structural attributes. The process starts by randomly generating a candidate merging recipe, combining a set of simple augmentations, such as mirroring, rotation, and scaling, with a pair of connection points selected from the guest and host graphs, shown as red points in (c,d). To assess the result produced by the recipe, we apply the augmentations and connect the glyphs at the selected points. We approve only recipes that produce a clean, coherent shape (e). A merging recipe is retained only if it succeeds across a minimum number of styles. Additional implementation details are provided in the appendix. The final merging result is shown in (f).

![](images/b3a148a1ad341e379364f061c7b0656f7bf57ca1e7141b261595dfe39c696015.jpg)  
Figure 7. Merging. The steps of merging the same pair of glyphs in two styles. Columns labelled as referenced in Sec. 3.2.

## 3.3. Style LoRA

We train a separate Style LoRA on sketch-glyph pairs for each target style, as shown in Fig. 3(iii). The backbone and UGL parameters remain fixed, while UGL continues to guide the geometry throughout training. Only the Style LoRA is optimized, allowing it to focus on how that geometry is realized in the target style. Here we use a low-rank adapter, far smaller than UGL: its limited capacity helps keep the style update from re-learning geometry or competing with the shared geometry prior. Following previous work [12, 23], we associate each style with a unique rare token. This token and a short appearance description provide the textual conditioning during both Style LoRA training and inference.

Geometry reinforcement. The few style examples also share a limited family of geometries, so the Style LoRA may still tie the learned style to those geometries despite UGL’s structural guidance. To preserve open-form transfer, we mix geometry-enriched sketch-glyph pairs into its training batches, shown by the connection from GEE to Style LoRA training in Fig. 3. These pairs expose the model to geometries outside the target style’s examples throughout adaptation, reinforcing its ability to follow new structures while learning the style.

## 3.4. Inference

After training, we apply UGL and the selected Style LoRA jointly to the base diffusion model to map structural sketches to styled outputs (Fig. 3 iv). To control geometry adherence and style strength separately, we adopt dual classifier-free guidance following InstructPix2Pix [3]. During training, the structural sketch, the style caption, or both are dropped to their null conditions, so a single network learns all four conditioning combinations. During sampling, we combine three forward passes using two CFG coefficients to obtain separate geometry and style guidance strengths, balancing adherence to the input sketch geometry vs. the target style. Appendix B provides the guidance formulation and implementation details.

This joint application supports transfer to new forms without further training: a pictogram style can depict new human poses, while a sculptural style can turn free-form strokes into tubular shapes (Fig. 1). The same sketch can also be rendered in different learned styles (Fig. 2). Section 4 evaluates these capabilities through qualitative and quantitative comparisons and ablations of the method’s main components.

## 4. Experiments

We evaluate our method across fonts, pictograms, and object-centric styles, using sketches of increasing structural novelty:

• In-domain, held-out test examples from the same source domain as the style training data

• Near-domain, semantically related but structurally unseen examples, such as non-Latin glyphs for fonts and everyday human poses for pictograms

• Out-of-domain, structures from unrelated shape families, such as non-glyph sketches for font styles or nonhuman sketches for pictogram styles.

We compare against prior methods and perform targeted ablations to assess both open-set transfer quality and the contributions of our main design choices.

## 4.1. Training and Evaluation Data

The training data for the universal geometry LoRA (UGL) is produced by the geometry enrichment engine (GEE) from 80 base glyphs (Latin letters, Latin Extended letters, and digits) shared across 16 fonts. Applying 6,000 random recipes yields 42,705 sketch–glyph pairs, on which UGL is trained once (Appendix D). Each style is represented by a small exemplar set with manually drawn structural sketches. We hold out six examples per style for evaluation and train the Style LoRA on the remainder, yielding roughly 20–70 sketch–exemplar pairs per style. During Style LoRA training, 10% of each batch is replayed from the UGL training pairs using their original style-free captions, helping preserve the learned geometry prior while adapting to style. For evaluation, we use a benchmark of 50 curated sketches per style category (fonts and pictograms). To analyze how well our method and the baselines generalize, the sketches span the in-, near-, and out-of-domain tiers defined above. We evaluate across twelve styles, and for each style all methods receive the same sketches and style references.

## 4.2. Open-form Geometric Generalization

We tailored our main technical contributions, the universal geometry LoRA (UGL) and the geometry enrichment engine (GEE), to open-form generalization and the challenges of fine-tuning large models.

In this section, we demonstrate the necessity of both components; later in Table 1 we also present a quantitative evaluation that supports each claim.

We begin with the most direct alternative: avoid the UGL by fine-tuning a single low-rank adapter on the few sketchglyph pairs available for a given style. While this formulation preserves both the target geometry and the learned style on held-out examples from the same style family, it does not generalize to out-of-domain structures. Instead, the model takes a shortcut by producing results resembling the closest training example, best seen in the bottom rows of Fig. 8 (w/o UGL), where the model outputs styled examples similar to English alphabet letters rather than the desired geometry. Row w/o UGL in Table 1 quantifies this behavior.

A natural hypothesis for this phenomenon is that each geometry is observed only through a single realization in the target style, providing too little variation to disentangle the geometry and style.

An intuitive solution is to train a geometry adapter on ordinary glyphs drawn from a broad corpus of fonts, using the same training settings as UGL but without employing GEE, and then train the same Style LoRAs on top of it. Unfortunately, the failure persisted, now in an even more revealing form. The UGL is even more biased towards common geometries from the train set, e.g. alphabet letters, as again seen in the bottom row of the w/o GEE section of Fig 8. These results are again supported by data in Table 1.

![](images/f3a5f4a54a4130c7b77c5ce92906435c88a23b39e838da239e32323529ef2107.jpg)

![](images/f252a20b634ef67af99099be4e7957164019129bd58269f6111f8afdfa144042.jpg)

![](images/a315a74009db82fde07766daba9bea169dfa983daf9a1ca5bc5123579a3b972a.jpg)

![](images/dbeb9db4acfeb7d4591875ee011737cd6c0171b513abb348f04af3cbdcc8efd9.jpg)  
Figure 8. Path to geometric generalization. Each pair shows an unseen sketch and the output in different styles. w/o UGL trains the Style LoRA without UGL, and w/o GEE trains the UGL without GEE. In both, the failure patterns of the geometry are marked. Ours follows the same sketches while preserving each style.

What is instead needed is a training distribution with fewer geometric correlations, preventing the model from capturing a specific shape identity. This is solved by our second technical contribution (GEE Sec. 3.2). The GEE’s immense augmentation strategy forces the geometry model to rely on the supplied structure rather than recognizing a known geometry. With this enriched corpus, completing our full method, the out-of-domain score rises to 0.94 (Ours in Table 1), with a geometry factor of 0.97 and style within

![](images/1bd20838b193587b40f1c15c1edaf1ca33c6efcb90016d51db59ba5fddb095a4.jpg)  
Figure 9. In-domain comparisons. Each row shows a sketch held out from that style’s training examples, two reference examples of the style, the outputs of the baselines, the ground truth, and our output. The font styles (top) are compared with the dedicated fontgeneration models, the colorful styles (bottom) with the generalpurpose image models.

3% of the ceiling, and the characteristic substitutions largely disappear.

## 4.3. Qualitative Evaluation

We compare against FontDiffuser [29], MX-Font [22], Gemini 3.1 (Nano-Banana 2) [14], and gpt-image-1- mini [21]. All methods receive the same target sketch and style references, except FontDiffuser, which uses a single reference. We include the general-purpose models for completeness, as they require no per-style training. Additional comparisons, including the adapter-based methods B-LoRA and ZipLoRA as well as AiS, are provided in Appendix A.

In the in-domain tier, we evaluate held-out sketches (Fig. 9). Our method closely matches the ground truth while preserving the requested geometry and fine stylistic details. Gemini and GPT often capture the overall style but drift toward familiar poses or glyphs and lose fine font details. FontDiffuser often deviates from the input structure, while MX-Font follows it more closely but transfers the target style less faithfully.

In the near-domain tier (Fig. 11), we transfer styles to related but unseen forms: non-Latin glyphs. Baselines often distort these unfamiliar inputs toward known forms or lose structural or stylistic details, while our method follows the supplied geometry and preserves the target style. The zoomed examples highlight reference motifs retained in the outputs, demonstrating transfer beyond the forms seen during style adaptation.

The out-of-domain tier (Fig. 10) provides a stronger test by applying pictogram and object-centric styles to structures from unrelated shape families. Across these sketches, our method retains the characteristic shapes and appearance of each style while following the new structure, demonstrating open-form transfer beyond the source domain.

## 4.4. Quantitative Results

We evaluate whether our method follows the structural sketch while realizing the target style, and how well both hold as the target geometry moves away from the style’s source domain. On the in-domain tier, where paired ground truth is available, we report LPIPS for perceptual similarity to the ground truth and SSIM for agreement in local image structure. For the near- and out-of-domain tiers, where paired ground truth is unavailable, we use VLM-asa-judge [4]. The judge separately scores structural adherence to the sketch and stylistic fidelity to the references; we report their product, normalized by the score of real examples from the corresponding style (judge setup and prompt in Appendix F).

![](images/fb46982d608e005dc17f0a4058e732109e2b635964cb082e3644773121d2d636.jpg)  
Figure 10. Out-of-domain results. Each result is shown directly below its framed structural sketch. Each sketch lies outside the shape family used to learn the corresponding style. Top row: a single sculptural style applied to glyphs from unseen scripts, a bridge, and free-form shapes. Reference examples are included to illustrate the diversity within the learned style. Bottom row: three additional styles: two pictogram styles (left & right) and a second sculptural style (middle)

![](images/f75945500331fb01297b5ed11dd188618f17317b850d9bab0c10a44a0238f0d0.jpg)  
Figure 11. Near-domain results. Each row shows a non-Latin glyph sketch, two reference examples of the style, the outputs of the baselines, and our output next to an enlarged view of the region marked in red. The font styles (top) are compared with the dedicated font-generation models, the colorful styles (bottom) with the general-purpose image models.

Comparison and ablation. Table 1 reports judge scores by style family (Fonts, Non-Fonts) and by distance from the training domain (Near, Out), followed by LPIPS and SSIM. Our full method achieves the best results across all reported measures. In the upper block, the font generators closely follow the sketch but lack style fidelity, whereas the generalpurpose models better capture the target style but drift from the input structure, as reflected in their lower Out scores. The lower block ablates our method. w/o UGL removes the UGL and trains the Style LoRA directly on the base model. Non-hierarchical trains the Style LoRA without the UGL and adds the UGL only at inference. w/o GEE trains the UGL, and applies geometry reinforcement, on original glyphs only, without GEE. w/o Reinforcement uses the GEE-trained UGL but omits geometry reinforcement during style training. Their performance shows that the importance of our design choices becomes more pronounced as the target geometry moves farther from the training domain. In particular, removing geometry reinforcement lowers the Near score from 0.975 to 0.966, but causes a substantially larger drop in the Out score, from 0.941 to 0.872. This suggests that preserving geometric adherence during style adaptation is particularly important for unfamiliar forms.

User study. We complement these measures with user preferences collected from 32 participants, who compare our result against each baseline in terms of style, structure, and overall preference (Appendix G). Fig. 12 shows the same trend as Table 1: our results are preferred over every baseline on all three criteria, with the largest margins in style and overall preference.

<table><tr><td></td><td colspan="4">VLM-as-a-judge↑</td><td colspan="2">LPIPS↓</td><td colspan="2">SSIM↑</td></tr><tr><td></td><td colspan="2">Per-style</td><td colspan="2">Distance from domain</td><td colspan="2"></td><td colspan="2"></td></tr><tr><td>Method</td><td>Fonts</td><td>Non-Fonts</td><td>Near-domain</td><td>Out-of-domain</td><td>Fonts</td><td>Non-Fonts</td><td>Fonts</td><td>Non-Fonts</td></tr><tr><td>FontDiffuser</td><td>0.459</td><td>NA</td><td>0.469</td><td>0.433</td><td>0.425</td><td>NA</td><td>0.473</td><td>NA</td></tr><tr><td>MX-Font</td><td>0.597</td><td>NA</td><td>0.627</td><td>0.522</td><td>0.315</td><td>NA</td><td>0.508</td><td>NA</td></tr><tr><td>GPT-Image-1-mini</td><td>0.773</td><td>0.713</td><td>0.766</td><td>0.796</td><td>0.299</td><td>0.304</td><td>0.473</td><td>0.326</td></tr><tr><td>Gemini (Nano-Banana 2)</td><td>0.918</td><td>0.971</td><td>0.936</td><td>0.927</td><td>0.284</td><td>0.239</td><td>0.505</td><td>0.433</td></tr><tr><td>w/o UGL</td><td>0.670</td><td>0.840</td><td>0.862</td><td>0.803</td><td>0.249</td><td>0.174</td><td>0.460</td><td>0.485</td></tr><tr><td>Non-hierarchical</td><td>0.915</td><td>0.763</td><td>0.738</td><td>0.735</td><td>0.250</td><td>0.264</td><td>0.465</td><td>0.382</td></tr><tr><td>w/o GEE</td><td>0.609</td><td>0.746</td><td>0.822</td><td>0.730</td><td>0.218</td><td>0.204</td><td>0.518</td><td>0.406</td></tr><tr><td>w/o Reinforcement</td><td>0.913</td><td>0.907</td><td>0.966</td><td>0.872</td><td>0.215</td><td>0.199</td><td>0.527</td><td>0.469</td></tr><tr><td>Ours</td><td>0.966</td><td>0.988</td><td>0.975</td><td>0.941</td><td>0.207</td><td>0.145</td><td>0.560</td><td>0.535</td></tr></table>

![](images/ebc17c2f08b158039469e7cedfc147b6d227fdda1d7e3ef27bf6ba032aae07c0.jpg)  
Table 1. Quantitative comparison and ablation. Our method most closely matches the ground truth according to LPIPS and SSIM, and achieves the highest VLM-as-ajudge scores on both font and non-font styles, across structures near to and far from the training domain. The ablations shown below degrade geometric fidelity, style fidelity, or both, with the largest performance drops occurring out of domain.  
Figure 12. Human preference study. Win rate of our method against each baseline. Our method dominates in both structure and style across all comparisons and consistently outperforms the baselines in overall preference.

## 5. Conclusions, Limitations and Future Work

We introduced Open-Form Geometric Style Transfer, where a geometric style learned from one family of shapes is applied to structures well beyond that family. We showed that learning geometric style requires separating how a shape is realized from the particular geometry on which that style is observed. However, disentanglement is necessary but not sufficient. To transfer a geometric language to unseen forms, the geometry component must itself be universal.

These findings suggest an asymmetric view of geometry and style. Geometry should be learned broadly, across a large and diverse space of structures, and can be learned once as a reusable capability. Style, in contrast, can be learned from a much narrower collection of examples and specialized independently. Our Universal Geometry LoRA and Style LoRA instantiate this separation, while GEE provides the geometric diversity needed to make the former universal.

As with many fine-tuned generative models, some challenges remain. Complex structures can expose a tradeoff between geometric and stylistic fidelity, while strong or recurring stylistic elements can occasionally interfere with the target structure or become overemphasized. Improving this separation, particularly for increasingly complex geometries and styles, is an interesting direction for future work. We believe this principle may extend beyond geometric style transfer, enabling learned visual languages to be applied outside the domains in which they were observed.

## References

[1] Daniel Berio, Frederic Fol Leymarie, Paul Asente, and Jose Echevarria. Strokestyles: Stroke-based segmentation and stylization of fonts. ACM Transactions on Graphics (TOG), 41(3):1–21, 2022. 3

[2] Black Forest Labs. Flux.2 [klein]: Compact flow models for interactive visual intelligence. https://huggingface. co/black- forest- labs/FLUX.2- klein- 4B, 2026. 16

[3] Tim Brooks, Aleksander Holynski, and Alexei A Efros. In structpix2pix: Learning to follow image editing instructions. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18392–18402. IEEE, 2023. 4, 7, 13

[4] Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. MLLM-as-a-judge: Assessing multimodal LLM-as-a-judge with vision-language benchmark. In Proceedings of the 41st International Conference on Machine Learning, pages 6562–6595. PMLR, 2024. 9

[5] Changwoon Choi, Hyunsoo Lee, Clément Jambon, Yael Vinker, and Young Min Kim. Image-guided geometric stylization of 3d meshes, 2026. 3

[6] Terrance DeVries and Graham W Taylor. Improved regular ization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552, 2017. 5

[7] Alexei A. Efros and William T. Freeman. Image quilting for texture synthesis and transfer. In Proceedings of the 28th Annual Conference on Computer Graphics and Interactive Techniques, page 341–346, New York, NY, USA, 2001. Association for Computing Machinery. 1, 3

[8] Alexei A Efros and Thomas K Leung. Texture synthesis by non-parametric sampling. In Proceedings of the seventh IEEE international conference on computer vision, pages 1033–1038. IEEE, 1999. 1, 3

[9] William T Freeman, Joshua B Tenenbaum, and Egon C Pasztor. Learning style translation for the lines of a drawing.

ACM Transactions on Graphics (TOG), 22(1):33–46, 2003. 2, 3

[10] Yarden Frenkel, Yael Vinker, Ariel Shamir, and Daniel Cohen-Or. Implicit style-content separation using b-lora. In European Conference on Computer Vision, pages 181–198. Springer, 2024. 3, 12

[11] Thomas Funkhouser, Michael Kazhdan, Philip Shilane, Patrick Min, William Kiefer, Ayellet Tal, Szymon Rusinkiewicz, and David Dobkin. Modeling by example. ACM transactions on graphics (TOG), 23(3):652–663, 2004. 5

[12] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit Haim Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In The Eleventh International Conference on Learning Representations, 2023. 7

[13] Leon A Gatys, Alexander S Ecker, and Matthias Bethge. A neural algorithm of artistic style. arXiv preprint arXiv:1508.06576, 2015. 2, 3

[14] Google DeepMind. Nano banana 2 (gemini 3.1 flash image). https://blog.google/innovation-andai/technology/ai/nano-banana-2/, 2026. Accessed September 2026. 8

[15] Amir Hertz, Andrey Voynov, Shlomi Fruchter, and Daniel Cohen-Or. Style aligned image generation via shared attention. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4775–4785. Ieee, 2024. 2, 3

[16] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. In NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications, 2021. 13

[17] Edward J Hu, yelong shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. 3, 16

[18] Zheyu Li, Honglie Wang, Yan-Ming Zhang, and Cheng-Lin Liu. Flux-font: stylized font generation based on fine-tuning pre-trained diffusion transformer. In Asian Conference on Pattern Recognition, pages 324–338. Springer, 2025. 2, 3

[19] Min Lu, Yuanfeng He, Anthony Chen, Jianhuang He, Pu Wang, Daniel Cohen-Or, and Hui Huang. Abstraction in style: Beyond texture and color. In Proceedings of the Special Interest Group on Computer Graphics and Interactive Techniques Conference Conference Papers, pages 1–11, 2026. 3, 12

[20] Chong Mou, Xintao Wang, Liangbin Xie, Yanze Wu, Jian Zhang, Zhongang Qi, and Ying Shan. T2i-adapter: Learning adapters to dig out more controllable ability for text-to-image diffusion models. In Proceedings ofthe AAAI conference on artificial intelligence, pages 4296–4304, 2024. 4

[21] OpenAI. Gpt image 1 mini. https://developers. openai.com/api/docs/models/gpt-image-1- mini, 2025. Accessed September 2026. 8

[22] Song Park, Sanghyuk Chun, Junbum Cha, Bado Lee, and Hyunjung Shim. Multiple heads are better than one: Fewshot font generation with multiple localized experts. In

2021 IEEE/CVF International Conference on Computer Vision (ICCV), pages 13880–13889. IEEE, 2021. 2, 3, 8

[23] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In 2023 IEEE/CVF Conference on Computer Vi sion and Pattern Recognition (CVPR), pages 22500–22510. IEEE, 2023. 7

[24] Viraj Shah, Nataniel Ruiz, Forrester Cole, Erika Lu, Svetlana Lazebnik, Yuanzhen Li, and Varun Jampani. Ziplora: Any subject in any style by effectively merging loras. In European Conference on Computer Vision, pages 422–438. Springer, 2024. 3, 12

[25] Joshua B Tenenbaum and William T Freeman. Separating style and content with bilinear models. Neural computation, 12(6):1247–1283, 2000. 2, 3

[26] Vikas Thamizharasan, Difan Liu, Shantanu Agarwal, Matthew Fisher, Michaël Gharbi, Oliver Wang, Alec Jacobson, and Evangelos Kalogerakis. Vecfusion: Vector font generation with diffusion. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 7943–7952. IEEE, 2024. 2, 3

[27] Andrey Voynov, Kfir Aberman, and Daniel Cohen-Or. Sketch-guided text-to-image diffusion models. In ACM SIG-GRAPH 2023 conference proceedings, pages 1–11, 2023. 2, 4

[28] Haofan Wang, Matteo Spinelli, Qixun Wang, Xu Bai, Zekui Qin, and Anthony Chen. Instantstyle: Free lunch towards style-preserving in text-to-image generation. arXiv preprint arXiv:2404.02733, 2024. 3

[29] Zhenhua Yang, Dezhi Peng, Yuxin Kong, Yuyi Zhang, Cong Yao, and Lianwen Jin. Fontdiffuser: One-shot font genera tion via denoising diffusion with multi-scale content aggregation and style contrastive learning. In Proceedings of the AAAI conference on artificial intelligence, pages 6603–6611, 2024. 2, 3, 8

[30] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In Proceedings of the IEEE/CVF International Con ference on Computer Vision (ICCV), 2019. 5

[31] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In 2023 IEEE/CVF International Conference on Computer Vi sion (ICCV), pages 3813–3824. IEEE, 2023. 4

[32] Li Zhao, Shan Dong, Jiayi Liu, Xijin Zhang, Xiaojiao Gao, and Xiaojun Wu. Skeleton-guided diffusion for font genera tion. Electronics, 14(19):3932, 2025. 2, 3

## A. Further Comparisons

Sec. 4 compares against font-generation models and general-purpose image generators. Here we include additional comparisons with three recent methods that are particularly relevant to our setting. B-LoRA [10] and ZipLoRA [24], like ours, use low-rank adapters to separate what is depicted from how it is rendered, while Abstraction in Style (AiS) [19] explicitly models structural abstraction as part of stylization. Fig. 13 shows the B-LoRA and ZipLoRA results next to ours for the same sketch and style; AiS is discussed separately below.

B-LoRA. B-LoRA leverages LoRA to implicitly separate the style and content of a single image by jointly learning the LoRA weights of two specific SDXL blocks, which can then be used as independent components. We use the official implementation and settings, which we verified by reproducing the authors’ published example. The content adapter is trained on the structural sketch and the style adapter on six training examples of the style, matching the other reference-based baselines. As Fig. 13 (left) shows, the style block carries only the palette and texture of the style: the geometric style is lost, and the structure of the sketch is replaced by hallucinated, unrelated content, because a thin line drawing provides little reconstruction signal.

ZipLoRA. ZipLoRA merges independently trained style and subject LoRAs by learning per-column merger coefficients that minimize their interference. As no official implementation exists, we implement it on our own backbone, which isolates the merging strategy: the content adapter is our UGL and the style adapter is a Style LoRA trained without the UGL, as in the Non-hierarchical ablation (Sec. 4.4). Both adapters are frozen and only the coefficients are optimized, with the paper’s hyperparameters. With all coefficients at one, the two adapters are simply added, which is exactly the Non-hierarchical configuration of our ablation study; sampling uses its sketches, prompts, seeds and guidance. The learned coefficients scale down both adapters, averaging 0.39-0.48 for the UGL and 0.24-0.52 for the Style LoRA across the three styles. As Fig. 13 (right) shows, the result follows the sketch, but the conflict between structure and style remains, only in a milder form: style-specific details such as the pictogram tile and the wedge-shaped terminals of the display font are weakened. Training the Style LoRA with the UGL active, as we do, avoids the conflict altogether.

Abstraction in Style (AiS). AiS is designed specifically for styles in which abstraction extends beyond color and texture to geometric and structural deformation. AiS explicitly separates structural abstraction from visual stylization in two stages: it first transforms the target into an intermediate abstraction proxy that reflects the structural tendencies of the reference style, and then renders this proxy into the final stylized output. Both stages are learned from a small set of visual exemplars through its Visual Analogy Transfer framework. This makes AiS particularly relevant to our setting, as it is one of the few recent methods that explicitly treats structural reinterpretation as part of style transfer.

![](images/a78122225434e79a17002944e77cf5e738c00fde080c6eb0bb60c06f9ccbb44e.jpg)  
Figure 14. Comparison with AiS. For each framed sketch, the result of AiS [19] (top) and ours (bottom), for two pictogram styles and a sculptural style.

![](images/580c0564871ef6aedd5c6209fe93628648111880f3cbadddb23386b7f0c823bb.jpg)  
Figure 13. Further comparisons. Each row shows the structural sketch, three examples of the style (framed), the result of the compared method, and ours. Left: B-LoRA [10] transfers palette and texture but not the structure. Right: ZipLoRA [24], applied to our UGL and a Style LoRA trained without it, follows the sketch but weakens the style.

Fig. 14 compares AiS with our method on three styles: two pictogram sets and a sculptural style. We train AiS on the same style references and use the structural sketch as the abstraction target. AiS captures the visual appearance of each style, but on the pictogram styles its outputs do not reliably follow the sketch: across the scripts, strokes are rearranged, omitted, or fused, while the runner pose is only loosely preserved. On the sculptural style, whose exemplars are themselves single continuous strokes, AiS follows the sketch more closely, though it renders every glyph as one uniform tube. This reflects AiS’s design: its abstraction proxy follows structural tendencies learned from the references rather than the geometry of a specific target. Our method follows each sketch closely while preserving the same style traits, including the tile deformation and stroke bands of the orange set.

## B. Dual Classifier-Free Guidance

Our generator is conditioned on two inputs: a sketch image $c _ { I }$ of the target geometry and a text caption $c _ { T }$ that carries the style token. Conventional text-conditioned classifierfree guidance (CFG) [16] has a single scale: it amplifies the caption, while the sketch enters both branches as an ordinary, unamplified condition, so the strength of the geometric conditioning cannot be controlled. We therefore adopt the compositional guidance of InstructPix2Pix [3], which separates the two conditions into terms with independent scales.

Null conditions. CFG requires the network to model the unconditional branches, so during training each sample is assigned to one of the four conditioning branches listed in Table 2: most samples keep both conditions, one in five drops the style caption, and a small share drops the sketch or both. The image null $\scriptstyle { \mathcal { O } } _ { I }$ is the complete absence of reference tokens, i.e. the model’s native text-to-image mode, rather than a blank or noise image, and the text null $\mathcal { D } _ { T }$ is the empty prompt.

<table><tr><td>Branch</td><td>% of samples</td><td>Geometry cI</td><td>Style cT</td></tr><tr><td>Fully cond.</td><td>70%</td><td>cI</td><td>CT</td></tr><tr><td>Geometry cond.</td><td>20%</td><td>cI</td><td> $\mathcal { D } _ { T }$ </td></tr><tr><td>Style cond.</td><td>5%</td><td> $\scriptstyle { \mathcal { Q } } _ { I }$ </td><td> $c _ { T }$ </td></tr><tr><td>Uncond.</td><td>5%</td><td> $\varnothing _ { I }$ </td><td> $\mathcal { O } _ { T }$ </td></tr></table>

Table 2. Conditioning branches during training and their sampling frequencies.

Guidance. Let $v _ { \theta } \big ( z _ { t } , c _ { I } , c _ { T } \big )$ denote the predicted velocity at noise level t. Each sampling step evaluates the unconditional, geometry-conditioned, and fully conditioned branches of Table 2 (the style-only branch is not used at sampling) and combines them as

![](images/a674f5a8cffc33c2f4b3675da67d7c9ee94f76b8272d2ddb920f37296f4778cf.jpg)  
Figure 15. The effect of the two guidance scales of Eq. 1; s<sub>T</sub> grows left to right, s top to bottom. The framed cell is our default operating point. The structural sketch and three reference examples of the style are shown below the grid.

$$
\begin{array} { r l } & { \widetilde { v } _ { \theta } ( z _ { t } , c _ { I } , c _ { T } ) \ = v _ { \theta } ( z _ { t } , \mathcal { O } _ { I } , \mathcal { O } _ { T } ) } \\ & { \quad \quad + s _ { I } \big [ v _ { \theta } ( z _ { t } , c _ { I } , \mathcal { O } _ { T } ) - v _ { \theta } ( z _ { t } , \mathcal { O } _ { I } , \mathcal { O } _ { T } ) \big ] } \\ & { \quad \quad \quad + s _ { T } \big [ v _ { \theta } ( z _ { t } , c _ { I } , c _ { T } ) - v _ { \theta } ( z _ { t } , c _ { I } , \mathcal { O } _ { T } ) \big ] . } \end{array}\tag{1}
$$

The first bracket moves the prediction toward the sketch and is scaled by the geometry scale $s _ { I } ;$ the second adds the contribution of the style token on top of the sketch and is scaled by the style scale $s _ { T }$ . With $s _ { I } = s _ { T }$ , the two brackets merge into one and the rule becomes single-scale CFG on the joint condition $\left( c _ { I } , c _ { T } \right)$ . Values of $s _ { I }$ above 1 amplify the geometric conditioning independently of the style scale. In practice we use $s _ { I } \in [ 4 , 6 ]$ and $s _ { T } \in [ 2 , 3 ]$ (Sec. 4).

Reading the two levers. Fig. 15 jointly sweeps the two guidance scales for a held-out sketch of $\mathrm { ~ a ~ } ^ { 6 6 } 3 ^ { 5 }$ . The grid makes the roles of the two guidance signals and their characteristic failure modes clear.

With no sketch guidance $( s _ { I } = 0$ , top row), the sketch is effectively ignored: increasing $s _ { T }$ strengthens the style but does not recover the geometry of ${ \mathrm { \Large ~ a ~ } } ^ {  } 3 ^ {  } . { \mathrm { ~ \normalfont ~ A t ~ } } s _ { I } = 1 .$ , a coherent letterform emerges, but with incorrect geometry, and strong style guidance can further distort its shape.

At the other extreme, strong sketch guidance $( s _ { I } = 8 ,$ bottom row) reliably preserves the geometry but weakens the appearance signal. Increasing s<sub>T</sub> restores the red tubular style, though gradients and translucency remain muted. The intermediate regime $( s _ { I } = 6 $ , third row) provides the best balance, recovering the correct $\mathbf { \bar { \Psi } } ^ { 6 6 } 3 ^ { \mathfrak { s } }$ while preserving the inflated volume and warm gradient of the exemplars. We therefore choose our operating point in this regime.

## C. Geometry Enrichment Engine: Details

This appendix details the construction of the geometryenriched training corpus used to train UGL (Sec. 3.2).

Generating unfamiliar geometries. We begin with 80 base glyphs (Latin letters, Latin Extended letters, and digits) shared across 16 fonts and treat them as raw geometric material. We expand this initial vocabulary using two simple structural operations: component removal, which removes a structural part of a glyph, and merging, which combines components from two glyphs. The operations are applied iteratively, allowing newly generated shapes to participate in subsequent operations. As illustrated in Fig. 4 of Sec. 3.2, the resulting forms quickly depart from recognizable characters, producing a much broader distribution of coherent but unfamiliar geometries.

We represent each sequence of structural operations as a geometric recipe. Importantly, a recipe is defined independently of a particular font and can therefore be applied to corresponding glyphs across multiple fonts. This produces related geometric constructions realized in different styles. Beyond increasing geometric diversity, this correspondence is what allows us to separate the structural information conveyed by a sketch from the style of its target realization.

A GEE recipe is a sequence of component-removal and merging operations, with both operations applied one to three times. The output of one step is the input of the next. Both operations share a first step, graph extraction, described next.

Graph extraction. Editing a glyph’s outline directly is fragile, since a removal or a join depends on local contour details that differ from style to style. Both operators therefore act on the glyph’s underlying stroke structure, which we extract as a graph; Fig. 5 of Sec. 3.2 illustrates the steps. Starting from the glyph (a), we compute its medial axis transform, which gives the centerline of every stroke together with the radius of the maximal inscribed disk at each point, a local estimate of stroke thickness (b). Treating the axis pixels as a graph, we prune the short branches caused by serifs and other stylistic contour details, retaining the dominant stroke structure (c). Chains of pixels between meeting points are then collapsed into single edges, junctions become nodes and stroke ends become tips (d). Finally, each edge is classified (e). An edge between two junctions is a bar, and an edge from a junction to a tip is a leg. Because these roles do not depend on the exact contour of a style, an operation on these graphs can be replayed across styles.

Structural correspondence. A recipe names the component it edits, but only within the style on which it was defined. To apply the same edit consistently across styles, we must identify the corresponding component in each style. We represent the selected component using a compact descriptor consisting of its type (bar or leg), the orientation of the chord connecting its endpoints, its midpoint normalized by the glyph’s bounding box, and its arc length relative to glyph size. For each other style, we compare this descriptor with all removable components of the same type and select the component with the smallest weighted distance, provided that the distance is below a matching tolerance. If no component falls below this tolerance, the recipe is not applied to that style. A recipe is retained only if at least a minimum number of styles remain (we configured as 6).

Component removal. Fig. 6 of Sec. 3.2 illustrates the process step by step. Given the extracted graph, we aim to remove exactly one structural component, such as a leg or bar, while keeping the rest of the outline unchanged and reconstructing the cut in the font’s own terminal style (a). The component to be removed is described by its structural attributes in the graph. In each style, we use these attributes to identify the corresponding component, so that the same component is removed consistently across styles. Removing only the skeleton associated with the component’s edge is not sufficient, since part of its medial axis may also belong to the junction. To avoid leaving an artificial bulge, we remove medial-axis points near the junction within a radius slightly smaller than its inscribed-disk radius (b). This creates a small gap, which we reconnect with a straight stroke that smoothly bridges the surviving branches. We remove pixels belonging to the deleted limb while preserving those shared with surviving strokes. We then fill small junction concavities using the convex hull of the surviving strokes (c). The removal is defined as a pixel mask, but the output must be vectorized. We therefore trace the removal region into a smooth contour and subtract it from the original glyph outline (d). Geometry outside the edited region is preserved exactly. The newly exposed cut boundary has no counterpart in the original glyph. To reconstruct it in a styleconsistent way, we search other glyphs from the same font for a matching outline segment, after normalizing for position, scale, and rotation. We then fit this donor segment to the exposed boundary and insert it into the cut region. The result is a glyph with one structural component removed while preserving the original outline wherever possible (e).

Merging. Fig. 7 of Sec. 3.2 illustrates the process. The connection between the two glyphs is described through structural attributes of their graphs. These attributes identify where the guest should attach to the host and allow the corresponding connection to be found across styles. We begin with two glyphs from the same style: a guest $g _ { a }$ (a) and a host $g _ { b } \ ( \mathsf { b } )$ . Our goal is to attach the guest to the host so that they form a single coherent glyph. The merge has two stages. First, we augment the pair,

$$
\hat { g } _ { a } = \mathrm { a u g m e n t } ( g _ { a } ) , \quad \hat { g } _ { b } = \mathrm { a u g m e n t } ( g _ { b } ) ,
$$

to generate multiple candidate configurations. The host may be mirrored, while the guest may be also rotated and scaled. These choices are randomly sampled when the recipe is created. In Fig. 7(c), for example, the host is unchanged while the guest is mirrored and rotated by 270<sup>◦</sup>. We then merge the augmented pair,

$$
\mathrm { m e r g e } ( \hat { g } _ { a } , \hat { g } _ { b } ) .
$$

On the guest, the connection is anchored at a stroke end facing the host, marked by the red dot in Fig. 7(c). On the host, we select the stroke end facing the guest; if none exists, we use the nearest skeleton point (Fig. 7(d)). The guest is then translated so that the two connection points align. We keep only merges that form a coherent single shape, with a localized overlap at the junction and no narrow gaps or small enclosed holes, as shown in Fig. 7(e). A merge recipe is retained only if it succeeds across a minimum number of styles, ensuring that the structural relation transfers consistently. Small fragments extending beyond the seam are removed to produce the final glyph, as shown in Fig. 7(f).

Style-suppressed structural sketches. For each generated glyph, we extract a structural sketch from its medial axis. A raw medial axis, however, can retain local characteristics of the font from which it was extracted, allowing the conditioning sketch to inadvertently reveal style. We therefore simplify the extracted structure by pruning small branches associated with local stylistic details and approximating the remaining structure with a limited set of Bézier curves. During training, we additionally apply random modifications to these curves. Together, these operations preserve the dominant structure while making residual stylistic cues unreliable.

We now describe the style-suppressing operations in more detail.

Pruning. We first remove branches that are more likely to arise from outline decoration than from genuine stroke structure. Pruning is defined relative to the local stroke width rather than by an absolute length threshold. In an initial pass, branches shorter than 0.4 stroke widths are removed. This suppresses small decorative spurs without systematically eliminating genuinely short structural components. Serifs require separate treatment. At a serif terminal, the medial axis often splits into two short branches on opposite sides of the stroke. Removing either branch independently would shorten the stroke rather than remove only its decoration. We distinguish such serif pairs from genuine bifurcations using the line segment joining the two branch tips. If this segment lies entirely inside the glyph, the branches are treated as the two prongs of a serif. If it exits the glyph, as between the upper arms of a Y, the branches are treated as genuine structure and retained. A detected serif pair is replaced by a single branch between the two prongs, preserving the full stroke extent while discarding the terminal shape. In our fonts, serif branches span at most approximately one stroke width, whereas genuine structural limbs typically span two to five stroke widths. We therefore merge a pair only when both branches are shorter than 1.5 stroke widths. A final pruning pass removes branches shorter than 0.9 stroke widths, except where such a branch independently captures a substantial part of the glyph structure.

Representational bottleneck. We next approximate the pruned medial axis with a limited number of cubic Bézier curves. This sparse representation acts as an explicit information bottleneck rather than merely a compact encoding. A limited number of smooth curves can preserve the coarse glyph geometry, including stroke paths, curvature, and connectivity, while lacking the capacity to reproduce fine stylistic irregularities inherited from the original outline. Curve fitting is performed adaptively. The axis is initially represented by a single cubic Bézier curve and is subdivided at the point of maximum fitting error only when needed. Capacity is therefore allocated preferentially to regions that require additional geometric detail. Across our data, a glyph is represented by approximately ten curves on average and by at most 12. If this capacity is exceeded, the least significant cues are discarded as intended. The subdivision tolerance is expressed relative to stroke thickness. Because the medial axis is defined by its distance to the glyph boundary, constraining the fitting error in this way keeps the approximation geometrically consistent with the glyph interior. The resulting representation is deliberately lossy while remaining faithful to the underlying stroke geometry.

Although pruning and curve compression remove much of the stylistic information, they cannot guarantee complete invariance. A sufficiently expressive model may still exploit residual correlations between the conditioning sketch and the target style. We therefore introduce two training-time mechanisms in UGL to make such residual information unreliable.

Perturbation during training. During training, each conditioning stroke is randomly perturbed while the corresponding target glyph remains unchanged. The model therefore does not observe a single exact skeleton as the unique conditioning representation of a target. Large-scale structure is preserved, while small local details become unreliable as predictive features. The curve representation allows us to perturb strokes structurally rather than at the pixel level. We deform the interior of each curve while keeping its endpoints fixed, preserving junctions and overall connectivity. Perturbation magnitude is scaled by curve length and correlated along the stroke, causing the stroke to bend smoothly as a whole rather than introducing highfrequency local noise. Each interior control point is displaced by a Gaussian whose standard deviation is a fraction of the curve’s chord length, capped at twice that deviation so that a curve cannot fold back on itself. We resample this fraction uniformly for every sample at every epoch, from [0.10, 0.20] when training UGL and from [0.03, 0.08] when training a Style LoRA (Table 3), so the model sees a range of sketch precision rather than one fixed noise level.

Cross-style pairing. To further prevent the sketch from serving as a style condition, we exploit the cross-font correspondence provided by the geometric recipes. Rather than always pairing a target glyph with its own extracted sketch, we can condition it on a sketch extracted from another font generated by the same recipe. Although their precise medial axes may differ, the sketch and target share the same underlying construction while belonging to different styles. This cross-style pairing discourages UGL from using the conditioning sketch to infer the target style.

Together, geometry enrichment, style suppression, and cross-style pairing produce a broad training distribution in which structural sketches primarily communicate geometry rather than shape identity or style. Training UGL on this corpus encourages it to follow the supplied structure, providing the universal geometry prior required for open-form transfer.

## D. Training details

Both adapters are LoRA modules [17] on the attention projections of the FLUX.2-klein-base-4B transformer [2] (80 linear layers); the backbone, the VAE and the Qwen3 text encoder stay frozen, the text encoder in 8-bit. UGL uses rank 192 with $\begin{array} { r } { \alpha \ = \ 1 9 2 ; } \end{array}$ ; every Style LoRA uses rank 32 with $\alpha \ = \ 3 2$ and is trained with UGL attached as a frozen adapter, never merged. Both stages use AdamW-8bit (learning rate $1 0 ^ { - 4 }$ , constant after a linear warm-up, $\beta = ( 0 . 9 , 0 . 9 9 9 )$ , weight decay 0.01, $\epsilon = 1 0 ^ { - 6 } )$ , an EMA of the adapter weights with decay 0.99, gradient checkpointing, and a flow-matching objective at $2 5 6 \times 2 5 6$ for both the target and the sketch condition. Adapter weights are stored in fp16. Table 3 lists the per-stage settings.

<table><tr><td></td><td>UGL (stage 1)</td><td>Style LoRA (stage 2)</td></tr><tr><td>LoRA rank / α</td><td>192 / 192</td><td>32 / 32</td></tr><tr><td>Optimisation steps</td><td>6,000</td><td>2,500</td></tr><tr><td>Batch × grad. accum.</td><td>16 × 4</td><td>4 × 2</td></tr><tr><td>Warm-up steps</td><td>400</td><td>50</td></tr><tr><td>Learning rate</td><td>10-4</td><td>10-4</td></tr><tr><td>Held out</td><td>5% recipes</td><td>6 rows per style</td></tr><tr><td>UGL replay in batch</td><td></td><td>10%</td></tr><tr><td>Sketch perturbation</td><td>10-20%</td><td>3-8%</td></tr><tr><td>Hardware</td><td>1× H100</td><td>1× L40S</td></tr></table>

Table 3. Training settings of the two stages.

![](images/46b52ed27267c77a39ffb0a830191946b7ef0df0562eb4683816b356132f9023.jpg)

![](images/ffad36bdb06d6801ee504c682f60247b17a4572d328a58271c96f24c923ccd09.jpg)

![](images/1028ba9c914d02b7a37c5937e455991c8e94462fed73f3eaad53b4cbac2bc6f7.jpg)

E. Additional Results

# ERS X X

Figure 16. Sculptural and decorative styles on pose and glyph sketches. Each row is one structural sketch (left): four pose sketches (top) and five glyph sketches (bottom); each column is one style, shown by three reference examples at the top.

![](images/2c237aa244472a65ed0d9489d4662de57c71cbb1c40cb91af56309c0d538f36f.jpg)  
Figure 17. Additional results. Each row shows three examples of a style (left) and our results for unseen structural sketches, each shown above its result. The rows cover font and pictogram styles.

![](images/a556c82a5abb73c20d7d3ab8adc22e0cf4c4002b19dda728cf8e3bd37f98b35f.jpg)  
Figure 18. Additional sculptural results. Two sculptural styles applied to unseen sketches. Each result is shown under its structura sketch.

![](images/09883736afb9b1b92f23dd68f6791085dc062f821ae95cca9485bcef9c935059.jpg)  
Open-Form Synthesis

## F. VLM-as-a-Judge Evaluation

We employ a VLM-as-a-Judge evaluation framework to assess generated results in the open-set regimes, where no unique paired ground truth exists: for a target structure outside the style’s own shape family there is no single correct rendering to compare against, so paired metrics such as LPIPS and SSIM are undefined. The judge is asked instead whether the requested structure is depicted and whether it is realized in the demonstrated geometric design language.

Each judged candidate is presented as a single composite image with labeled panels: a top row of K = 6 style references, and a bottom row holding the target structural sketch and the candidate. References are drawn only from the style’s training rows, so the judge never sees the groundtruth target of the case it is grading. All panels are rendered at 256 × 256 on a common canvas, and nothing in the image or the prompt identifies the method that produced the candidate.

We score each candidate absolutely rather than in pairs. This has two consequences we rely on: the resulting scalar does not depend on which baselines happen to be included in the comparison, and there is no left–right position bias to mitigate. To remove the remaining ordering effect, the six style references are shuffled per case with a fixed percase permutation. The judge is instructed to return a single JSON object conforming to a strict response schema, which prevents the answer from drifting into prose and bounds the output length.

The response to the example query illustrates the two axes being scored independently. Geometry is 4 rather than 5 because the judge locates two nameable departures from the sketch before scoring it: the rider is given the head the sketch leaves implicit, and the bicycle frame is redrawn cleaner than the tangle of strokes it was traced from. Style is 4 because the candidate carries the references’ badge, weight and figure convention onto a sport none of them shows, while stopping short of being mistaken for one of

## VLM-as-a-Judge: Evaluation Prompt (abridged)

System prompt: meticulous visual judge for a geometric style transfer benchmark; sees one composite image with labeled panels; answers only in schema-conforming JSON

## User prompt:

Image layout. Top row: 6 Style References, defining how things are drawn (strokes, curvature, corners, joins, color, texture, motifs). Bottom row: the Target Sketch, defining what must be depicted, and the Candidate.

Questions, answered in order:

1. geometry\_deviations: list every structural difference from the sketch, each tagged core (element missing, added, moved or reconnected) or minor (element slightly shifted or reshaped).

2. geometry\_score (1–5): is the sketch’s structure depicted? Deviations the references explain as style are allowed; style is not judged here. 5 = fully recognizable . . . 1 = a different object.

3. style\_score (1–5): does the candidate belong to the references’ visual family? Depicting the same objects is not rewarded; geometry is not judged here. 5 = could pass as a reference . . . 3 = several traits missing . . . 1 = no resemblance.

4. Flags: source\_leakage, the candidate reproduces a shown reference instead of the sketch; sketch\_copy, the sketch passed through with little or no style.

Output: one JSON object, {"geometry\_deviations": [{"what", "severity"}], "geometry\_score", "style\_score", "source\_leakage", "sketch\_copy"}.

## VLM-as-a-Judge: Example Query

Image input: one composite, sent alongside the prompt above. Here the style is a pictogram set and the requested structure is a cyclist: none of the six references depicts cycling, so the judge must separate the design language — white figure, dark badge, uniform stroke — from the particular sports that carry it.

![](images/4ddf167680d5c809858e56c975612f955c5b49cd5959768942281a9741362544.jpg)  
Response:

{"geometry\_deviations": [{"what": "added head", "severity": "core"}, {"what": "modified frame", "severity": "minor"}], "geometry\_score": 4, "style\_score": 4, "source\_leakage": false, "sketch\_copy": false}

them. Both flags are false: the candidate reproduces no shown reference, and it is not the sketch passed through unstyled.

## F.1. Judge Configuration

We use the same judge configuration for the entire evaluation: all reported numbers come from gemini-3.1-propreview with a structured response schema and medium media resolution. Each candidate is judged in a single pass. Scores are aggregated hierarchically, matching the rest of our protocol: samples within a case, cases within a style, then styles weighted equally.

## G. User Study

## G.1. User Study Setup

We complement the VLM-as-a-Judge evaluation with a human preference study. Unlike the judge, which scores each candidate absolutely, the study is pairwise: participants see two results for the same input and indicate which one is better. Relative judgments are easier for non-expert participants to make consistently than absolute ratings, and they directly answer the question of interest, namely whether our results are preferred over those of each baseline.

Each question is presented as a single composite image with labeled panels: a top row with the style references and the structural sketch, and a bottom row with two results, A and B. One result is ours and the other is produced by a baseline, both generated for the same style and the same sketch. All panels, including the two results, are rendered at the same size on a common canvas, so that the native output resolution of a method is not revealed, and nothing in the image or the text identifies the method behind a result.

The study contains 12 questions: three for each of the four baselines (MX-Font, FontDiffuser, Gemini, and GPT-Image), each on a different style, covering black-and-white fonts, decorative fonts, and pictogram styles. The sketches span held-out glyphs, non-Latin glyphs, human poses, and symbols. For each question, participants answer three multiple-choice items, on style, structure, and overall preference, each with the options Result A, Result B, and About the same.

To control for ordering and position effects, we prepare four editions of the questionnaire. Each edition shuffles the order of the questions, never placing the same baseline in two consecutive questions, and randomizes which side shows our result. Within every edition our result appears as A in exactly half of the questions, and the side assignment is mirrored between editions, so that any preference for a side cancels out. Participants reach the study through a single link that assigns them to one of the editions at random. Responses are anonymous, and each participant answers one edition.

## User Study: Instructions

## Thank you for participating!

This study evaluates geometric style transfer: the task of applying the design language shared by a few style references to a new, unseen structure given by an input sketch. The sketch specifies the target structure, whereas the references specify the style in which that structure should be realized.

Each of the 12 questions shows:

• the style references (target style)

• the structural sketch (target structure)

• two results, A and B, generated by different methods

For each question, choose which result (A, B, or about the same): • Style – better matches the style of the references.

• Structure – better depicts the structure of the sketch.

• Overall – you prefer overall.

Look closely at all the style references: a style is often defined by distinctive motifs that appear in only some of them.

## User Study: Example Question

Image shown: sport pictograms as the style and a treble clef as the structure (out of domain). Result A is ours and Result B is Gemini; other editions swap the sides.

![](images/4045234772c2186acfda4a2b524889aee494eeca9faa408a0515dcb18e674057.jpg)  
Items (Result A / Result B / About the same): Style, closer to the references? Structure, closer to the sketch? Overall, preferred?

The example illustrates why the two axes are asked separately. Both results depict a recognizable treble clef, so the structure item mainly asks whether the particular structure of the sketch is preserved, such as its proportions, the spiral around the stem, and the hooked tail. The style item asks something different: whether the result carries the distinctive traits of the references, such as the rounded, asymmetric blue tile, the bold white strokes with rounded ends, and the thin parallel lines that accompany the figures, rather than merely sharing their colors.

Before the first question, participants are shown one worked example in the same layout, with the intended answers marked and a one-line explanation of each. Its two results are not method outputs: one is a letter of a reference font and the other is the same letter in a plain serif font, so the intended answers are unambiguous and do not bias participants toward any method.

## G.2. Aggregation

For each baseline and each of the three items, we report the percentage of votes that prefer our result, prefer the baseline’s result, or rate the two as about the same, pooled over participants, editions, and the three questions of that baseline. Significance is assessed with a two-sided exact binomial test against 50% on the decisive votes, i.e., after discarding About the same answers. In total, 32 participants completed the study.