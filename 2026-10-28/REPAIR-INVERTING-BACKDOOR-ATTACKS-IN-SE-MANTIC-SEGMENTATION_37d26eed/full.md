# REPAIR: INVERTING BACKDOOR ATTACKS IN SE-MANTIC SEGMENTATION

Arnold Brosch Heinrich Heine University Dusseldorf¨ arnold.brosch@hhu.de

Kira Maag   
Heinrich Heine University Dusseldorf¨   
kira.maag@hhu.de

## ABSTRACT

Despite significant advances in computer vision, semantic segmentation models remain vulnerable to backdoor attacks, posing a security risk. These attacks redirect pixels of a chosen victim class to an attacker-specified target class when a trigger is present, while preserving clean predictions. Existing defenses are largely derived from image classification, while dedicated segmentation defenses remain scarce and typically require retraining or model modification. We introduce Re-PAIR, a post-hoc, label-free, attack-unaware defense based on the observation that backdoor-induced shifts can be stronger in the classifier head than in internal representations, leaving recoverable victim-class semantics. RePAIR infers the victim-target relation from an unlabeled, potentially contaminated image pool, constructs a safety-approved semantic reference, and localizes suspicious pixels by comparing head predictions with internal semantic evidence. It then applies feature-space and output corrections toward the inferred victim semantics, while conservative localization and output constraints limit changes to benign predictions. We model the attack as adversarial probability transport and repair as its approximate reversal, deriving bounds and sufficient conditions for localization, residual attack success, clean-side changes, defense-induced target predictions, and attack-unaware pair identification. Experiments across segmentation architectures and attack settings show that RePAIR substantially reduces attack success while largely preserving clean performance.

## 1 INTRODUCTION

Semantic segmentation assigns a semantic class to every pixel proving perception in safety-critical systems such as automated driving (Cakir et al., 2022) and medical imaging (Taghanaki et al., 2024). Semantic segmentation networks are vulnerable to backdoor attacks that preserve normal performance on benign inputs while inducing targeted failures when a trigger is present (Li et al., 2021b). For instance, a backdoor can cause a vehicle to disappear from the semantic map by relabeling its pixels as road, without affecting the prediction for the rest of the image. Semantic segmentation backdoors have developed from fixed trigger patterns and predefined victim–target mappings (Li et al., 2021b) toward increasingly flexible attack settings. Existing attacks allow the victim class to depend on trigger placement (Mao et al., 2023), exploit contextual relationships between semantic classes (Abbasi et al., 2025), or automatically optimize triggers and attack configurations (Lan et al., 2024). These attacks demonstrate that backdoors can induce targeted and spatially localized prediction changes under flexible conditions, making the development of robust defenses challenging.

While attacks on dense prediction tasks have received increasing attention, effective defenses remain comparatively underexplored. Detection-oriented methods can identify suspicious inputs or models but do not necessarily remove the underlying backdoor (Liu et al., 2023). Mitigation approaches based on retraining (e.g. fine-tuning and pruning) often require additional clean data and may alter the model’s behavior (Liu et al., 2018). For semantic segmentation, effective defenses should selectively correct attacked pixels while limiting changes to unrelated regions, motivating selective repair rather than global model modification.

In this paper, we introduce a Representation-guided pixel-wise attack Repair (RePAIR) against backdoor attacks on semantic segmentation models. We focus on a key empirical observation: the backdoor-induced shift can be substantially stronger in the final task decision than in an appropriately chosen internal semantic representation, even when the encoder itself is affected by the backdoor. Victim semantics can therefore remain partially recoverable even when the head predicts the attacker-selected target. The representation that exposes this state is architecture-dependent. E.g. for DeepLabV3+ (Chen et al., 2018) with a ResNet-50 backbone, the final backbone stage immedi ately before atrous spatial pyramid pooling provides a useful reference. For SegFormer (Xie et al., 2021) with a MiT-B5 backbone, no single block is sufficient. Victim evidence is distributed across several internal blocks and is reconstructed by a learned multi-layer logistic read-out.

![](images/d724c19d7171ac8fd5e51b4a7d8e1b04bc271adf2391f7e8be2000502ca918c4.jpg)  
Figure 1: From left to right: poisoned image, prediction under the backdoor attack, prediction after applying our defense and ground truth segmentation.

This observation has a natural geometric interpretation. For each pixel, the predicted class probabilities form a point in the probability simplex, where each coordinate represents the probability assigned to one semantic class. An idealized targeted backdoor shifts probability mass from a victim class to an attacker-selected target class, moving the prediction toward the target and away from the victim while remaining within the same simplex. Natural segmentation errors can also shift probability mass between classes (typically not concentrated in one direction), but a backdoor induces a systematic flow from one semantic class to another.

This directed view can be described by a local transport operator acting on the class-probability distribution. RePAIR approximates the corresponding reverse transport from surviving victim evidence and model-derived reference geometry, while treating feasible victim-target pairs as hypotheses and identifying the attacked relation without prior knowledge of either class.

A correct victim–target pair alone is not sufficient for a safe repair. The reference data used for repair must also be protected from poisoned samples, otherwise, the defense may learn the very distortion it is meant to remove. We therefore first identify candidate clean images and then check whether they provide reliable evidence of the victim class before using them for repair. An image is rejected if its internal victim evidence is severely degraded, or if a moderate degradation is accompanied by an unusually strong shift from the victim class to the target class. Rejected images are not replaced merely to meet a desired reference-set size. If too little reliable victim evidence remains, the defense does not construct a repair state. Together, these components form our method for selective and reliable backdoor repair. The source code of our method is publicly available at https:// github.com/A-Brosch/RePAIR. Our contributions can be summarized as follows:

• We introduce a post-hoc repair method for backdoored semantic segmentation models that reconstructs victim-consistent semantics and selectively corrects attacked pixels, without retraining or globally modifying the deployed network.

• We develop a theoretical framework that models targeted segmentation backdoors as directed victim-to-target transport in the class-probability simplex. This view links surviving internal evidence to localization, residual attack success, attack-unaware pair identification, and clean-side safety.

• Our method for generating a reliable repair reference is attack-unaware and operates on potentially compromised data. It identifies consistent victim-to-target evidence and screens candidate clean-like images with a one-way victim-safety firewall.

• We develop architecture-specific semantic references for both convolutional- and transformer-based segmentation models, showing that the information required for repair may be concentrated in a single representation or distributed across multiple internal layers.

## 2 RELATED WORK

Backdoors Attacks. The first work adapted the BadNets attack (Gu et al., 2019) from image classification to semantic segmentation, using either a black stripe at the image border as a non-semantic trigger or a semantic class present in the image as a trigger (Li et al., 2021b). Once activated, the attack relabeled all pixels of a victim class as a target class. The Influencer Backdoor Attack (IBA, (Lan et al., 2024)) places a trigger on non-victim pixels while relabeling the victim class as a predefined target. It further introduces Nearest Neighbor Injection (NNI) and Pixel Random Labeling (PRL), which respectively place the trigger near the victim region and add random label changes to non-victim pixels to strengthen the attack’s reliance on global context. The Object-Free Backdoor Attack (OFBA, (Mao et al., 2023)) places the trigger on the target object and varies victim classes and trigger positions during training, allowing the victim class to be selected by trigger placement at inference time while keeping the target class fixed. The Contextual Segmentation attack (ConSeg) uses a natural semantic trigger, such as a motorcycle, instead of an artificial pattern (Abbasi et al., 2025). It relabels the victim class as the target and adds a small number of co-occurring targetrelated classes within the victim region to strengthen the contextual association. BADSEG (Zhang et al., 2026) broadens the attack space to four attack types: Object-to-Object (O2O), Object-to-Background (O2B), Background-to-Object (B2O), and Background-to-Background (B2B), along with instance-level and conditional variants. A surrogate model optimizes trigger parameters and selects victim-target pairs using feature-space class centroids.

Defense Methods. Work on defenses against backdoor attacks in semantic segmentation remains limited, and existing studies therefore mainly adapt defenses originally developed for image classi fication. Test-time detection methods such as STRIP (Gao et al., 2019) and TeCo (Liu et al., 2023) detect suspicious inputs by measuring prediction entropy under image superposition or prediction consistency under image corruptions, respectively. Similarly, Zeng et al. (Zeng et al., 2021) detect anomalous trigger-related patterns in frequency domain (Ahmed et al., 1974). BEATRIX (Ma et al., 2023) detects anomalous internal representations using class-conditional higher-order Gram matrices (Zhang et al., 2026). These methods aim to identify suspicious inputs but do not repair the affected predictions or remove the underlying backdoor. Pruning (Liu et al., 2018) removes neurons with low activation on benign inputs and fine-tunes the remaining network on clean data to eliminate dormant backdoor components and recover clean performance. Fine-tuning (Sha et al., 2022) instead directly retrains the potentially compromised model on clean data to weaken the learned association between trigger and target. Both methods were also evaluated as defenses against segmentation backdoors, but both require access to clean data for mitigation (Lan et al., 2024; Zhang et al., 2026). Segment Cleanse (Mao et al., 2023) adapts Neural Cleanse (Wang et al., 2019) to semantic segmentation by reconstructing object-level trigger candidates and mitigating detected backdoors through retraining, likewise requiring model modification and clean data. Anti Backdoor Learning (ABL, (Li et al., 2021a)) exploits differences in training loss dynamics to identify suspicious samples and reduce their influence through gradient ascent. While these approaches identify or suppress backdoor behavior, they do not enable post-hoc, selective repair of affected pixels in a deployed model.

RePAIR differs from detection-based methods (e.g. STRIP, TeCo, frequency-domain defenses, and BEATRIX), which identify suspicious inputs, samples, or internal representations but do not repair the affected predictions. In contrast to model-modification approaches such as pruning, fine-tuning, and Segment Cleanse, our method does not retrain or otherwise alter the compromised model. It is also applied post-hoc, unlike ABL, which operates during training. Finally, RePAIR requires neither ground truth labels nor an external clean reference set, instead deriving the information needed for selective repair from the model and available data while leaving unaffected predictions unchanged.

## 3 THEORETICAL FRAMEWORK

We view a targeted backdoor in semantic segmentation as a trigger-induced, spatially selective change in dense predictions. Semantic segmentation predicts a probability distribution over semantic classes for each pixel, allowing a backdoor to selectively alter the predictions of pixels belonging to a particular victim class. During training, the attacker introduces triggered images with modified pixel-wise labels, assigning victim-class pixels to an attacker-selected target class. When activated, the backdoor causes pixels belonging to a victim class to be relabeled as an attacker-selected target class, while leaving other pixels largely unaffected. We treat the images as samples of a stochastic process, we further assume the neural network to learn the induced distribution during training. In order to change the argmax of the attacked pixels’ prediction the trigger has to lead to redistribution of probabilisitc mass from one atom to another (assumed to be mostly from victim to target class). The possibility of recognizing and “inverting” this transport yields two effects, the identification of victim and target class and nullyfication of the triggers effect, thus rendering it use- and harmless.

![](images/94b9facbd807f7b9560158f3e37bc7d6744fc45a7e751568eab927ea4bc68892.jpg)  
Figure 2: Targeted backdoor as directed victim-to-target transport. The prediction may move toward the target while internal representations retain victim evidence, which RePAIR uses for localization and repair. The forward operator and its coupling describe the same directed mass transfer at the marginal and joint-measure levels, respectively.

We model a targeted segmentation backdoor as a spatially selective victim-to-target distortion, corresponding to a directed redistribution of class probability from a victim class v to a target class $t \neq v$ . A successful backdoor may drive the final segmentation head toward the target class while an architecture-specific internal reference still retains evidence of the original victim semantics. This disagreement between the final prediction and the internal semantic evidence provides the signal used for localization and repair. RePAIR exploits this mismatch without assuming encoder invariance, semantic ground truth, trigger presence, or prior knowledge of the attacked pair.

Directed distortion. Let $p ^ { \star } \in \Delta ^ { K - 1 }$ denote the clean reference probability vector over K semantic classes and q its triggered counterpart predicted by a semantic segmentation network. The attack is modeled locally as a transfer of probability mass $\xi \ge 0$ from the victim class v toward the target class t

$$
q = T _ { \xi } ( p ^ { \star } ) = p ^ { \star } + \xi ( e _ { t } - e _ { v } ) , 0 \leq \xi \leq p _ { v } ^ { \star } ,\tag{1}
$$

where $e _ { c }$ is the unit vector associated with class c. The geometric interpretation shown is in Fig. 2. Thus the victim probability decreases by $\xi ,$ the target probability increases by $\xi ,$ and the victimover-target margin $\psi _ { v t } ( p ) = p _ { v } - p _ { t }$ satisfies

$$
\psi _ { v t } ( q ) = \psi _ { v t } ( p ^ { \star } ) - 2 \xi .\tag{2}
$$

Assuming $p _ { v } ^ { \star } > p _ { t } ^ { \star }$ , pairwise preference changes once $\xi > ( p _ { v } ^ { \star } - p _ { t } ^ { \star } ) / 2$ . A complete class change additionally requires the target coordinate to exceed all remaining classes. This is an analytical description of the dominant directed failure, not a claim that the poisoned network is globally onedimensional or that RePAIR observes the unknown clean $p ^ { \star }$ or the transported mass $\xi .$

Inverse operator and transport coupling. The local model in Equation equation 1 makes the notion of inverting precise. If ξ were known and q lay in the feasible image of $T _ { \xi }$ , the inverse marginal map would be

$$
T _ { \xi } ^ { - 1 } ( q ) = q + \xi ( e _ { v } - e _ { t } ) .\tag{3}
$$

The same forward displacement can also be represented by a one-edge coupling $\gamma _ { \xi }$ . With $( \gamma _ { \xi } ) _ { v v } =$ $p _ { v } ^ { \star } - \xi , ( \gamma _ { \xi } ) _ { v t } = \xi$ , and $( \gamma _ { \xi } ) _ { k k } = p _ { k } ^ { \star }$ for $k \neq v$ , its marginals satisfy

$$
\gamma _ { \xi } { \bf 1 } = p ^ { \star } , \gamma _ { \xi } ^ { \top } { \bf 1 } = q .\tag{4}
$$

These are related but distinct descriptions of reversal. The inverse operator $T _ { \xi } ^ { - 1 }$ maps the displaced marginal $q$ back toward $p ^ { \star }$ , whereas the transposed coupling $\gamma _ { \xi } ^ { \top }$ exchanges the forward coupling’s marginals and therefore represents the corresponding reverse mass flow as a joint measure. The transposed coupling is not itself the inverse operator, and no optimal-transport claim is implied without specifying a transport cost. RePAIR observes neither $\xi$ nor the hidden attack operator or coupling. Its repair instead approximates the reverse victim-directed transport from the discovered pair, surviving internal victim evidence, and safety-approved reference geometry. This is the sense in which RePAIR inverts the backdoor effect.

Surviving internal evidence and localization. Let $A = \{ \widehat { Y } ^ { \mathrm { h e a d } } = t \}$ denote successful baseline attack on a triggered pixel of the victim class, where $\widehat { Y } ^ { \mathrm { h e a d } }$ is the prediction of the undefended segmentation head. Let $\rho _ { v }$ indicate whether the internal representation still retains evidence for the victim class, and we define $\eta _ { \mathrm { r e f } }$ as the fraction of successful baseline attacks for which this direct victim evidence is absent. Thus, a small $\eta _ { \mathrm { r e f } }$ means that victim evidence survives in most successful attacks, although this does not guarantee correct recovery. When the released pair equals the true attacked pair, every successful attack with surviving victim evidence is selected by the direct semantic arm. Therefore,

$$
\eta _ { \mathrm { l o c } } \leq \eta _ { \mathrm { r e f } } .\tag{5}
$$

Additional residual branches can only reduce the miss rate further.

Residual attack success. Among triggered victim pixels, let $\varepsilon _ { \mathrm { r e p } }$ be the fraction of localized baseline attacks that still predict t after repair, and let $\delta _ { \mathrm { n e w } }$ denote defended target predictions outside the original baseline attack event. Then the defended attack success rate is bounded by

$$
\mathrm { A S R } _ { \mathrm { d e f } } \leq \mathrm { A S R } _ { \mathrm { b a s e } } \big [ \eta _ { \mathrm { l o c } } + ( 1 - \eta _ { \mathrm { l o c } } ) \varepsilon _ { \mathrm { r e p } } \big ] + \delta _ { \mathrm { n e w } } \ .\tag{6}
$$

The bound separates three failure modes: missed attacked pixels, unsuccessful repair on localized pixels, and new target predictions introduced elsewhere. When the released pair equals the true attacked pair, combining $\eta _ { \mathrm { l o c } } \leq \eta _ { \mathrm { r e f } }$ with the residual-ASR bound gives

$$
\mathrm { A S R } _ { \mathrm { d e f } } \leq \mathrm { A S R } _ { \mathrm { b a s e } } \big [ \eta _ { \mathrm { r e f } } + ( 1 - \eta _ { \mathrm { r e f } } ) \varepsilon _ { \mathrm { r e p } } \big ] + \delta _ { \mathrm { n e w } } \ .\tag{7}
$$

Thus, surviving internal victim evidence limits the fraction of successful attacks that can be missed by the localizer. Whether these localized attacks are actually recovered still depends on the subsequent repair step. The No New Target Veto prevents the defense from introducing unsupported target predictions outside the authorized repair support.

Attack-unaware pair identification. The victim and target classes are not assumed to be known. Instead, RePAIR evaluates all ordered class pairs using an attack-unaware image-level evidence score. For a pair (a, b), let $G _ { i } ( a , b )$ denote the score on image-level, and let $g _ { a b } = \mathbb { E } [ G _ { i } ( a , b ) ]$ be its expected value. For analysis, we decompose this score into natural evidence and an attackinduced contribution $g _ { a b } = \nu _ { a b } + \beta _ { a b }$ . Advantage over every competitor exceeds the strongest competing natural-evidence advantage plus $\delta _ { \mathrm { e s t } }$ The attacked pair can be uniquely identified if its attack-induced advantage over every competing pair is larger than both the strongest naturalevidence advantage of a competitor and the estimation error $\delta _ { \mathrm { e s t } }$ . This provides a sufficient separation condition for pair identification, but does not constitute a distribution-free guarantee for the implemented multi-route selector.

Clean-side safety. Let $\kappa _ { \mathrm { c l } }$ denote the probability that a benign pixel is selected for repair, and let $ { \delta _ { \mathrm { s p i l l } } } ^ { \mathrm { c l } }$ denote the probability of a defended prediction change outside the selected mask. Then

$$
\mathbb { P } _ { \mathrm { c l } } ( \widehat { Y } ^ { \mathrm { d e f } } \neq \widehat { Y } ^ { \mathrm { h e a d } } ) \le \kappa _ { \mathrm { c l } } + \delta _ { \mathrm { s p i l l } } ^ { \mathrm { c l } } ,\tag{8}
$$

where $\mathbb { P } _ { \mathrm { c l } }$ is the distribution over clean inputs and $\widehat { Y } ^ { \mathrm { d e f } }$ the prediction after repair. Thus, clean prediction changes are bounded by the probability of selecting benign pixels and the probability of changes outside the repair mask. Exact preservation is the special case where both terms are zero. The reference-calibration budgets are empirical constraints on the safety-approved population and do not provide population-level guarantees for unseen benign inputs. The framework does not require reconstruction of an unobserved clean class distribution. Instead, it uses surviving victim evidence, attack-unaware pair identification, conservative localization, and a safety-approved reference population to selectively repair target predictions. The forward transport and its reversal provide the conceptual model, while the implemented defense approximates this reversal using model-derived evidence. Complete definitions and proofs are provided in Appendix A.1.

## 4 METHOD

RePAIR is a post-hoc, label-free, attack-unaware defense for targeted semantic segmentation backdoors. The compromised model remains frozen. The defense uses neither trigger reconstruction nor retraining, and uses no semantic ground truth, clean/poison provenance, attack mask, attack family, poisoning rate, or predefined victim-target pair. Its inputs are the model and an unlabeled image pool that may contain triggered samples. The method has two stages. First, an offline builder identifies a directed victim-target relation from model-derived evidence and constructs a conservative, safety-approved reference state. Throughout this section, $( v , t )$ denotes the operational victim–target pair released by the attack-unaware builder. At inference, a pair-conditioned localizer identifies inconsistent pixels, feature transport moves them toward victim semantics, and output safety limit unintended changes. More details are provided in Appendix A.2.

This operational pipeline implements the reverse-transport interpretation of Section 3 without estimating the hidden attack operator $T _ { \xi }$ , its transported mass $\xi ,$ or the coupling $\gamma _ { \xi }$ . Pair discovery estimates the semantic direction, the safety-approved state supplies a victim-side reference geometry, and localization determines where that reverse direction may be applied. Repair magnitude is therefore pixel- and branch-dependent rather than an estimate of a single global ξ.

Architecture-specific semantic reference. We write the segmentation network as $f = h \circ g .$ , where g is the feature extractor and h the segmentation head. For each pixel $x ,$ the head produces class probabilities $q _ { k } ( x )$ and prediction $\widehat { Y } ^ { \mathrm { h e a d } } ( x ) = \arg \operatorname* { m a x } _ { k } q _ { k } ( x )$ . RePAIR compares this prediction with an architecture-specific internal semantic reference, which differs architectures.

For convolutional networks (CNNs), we use the normalized final backbone feature $z ( x )$ . For each class $k ,$ its reference centroid is computed as $\begin{array} { r } { \mu _ { k } = | \mathcal { P } _ { k } | ^ { - 1 } \sum _ { x \in \mathcal { P } _ { k } } z ( x ) } \end{array}$ , where $\mathcal { P } _ { k }$ contains pixels assigned to k by the frozen segmentation head. The internal prediction is then the class with the most similar centroid under cosine similarity $\widehat { Y } ^ { \mathrm { p r o t } } ( x ) = \arg \operatorname* { m a x } _ { k }$ sim $( z ( x ) , \mu _ { k } )$ ). Reference estimation is image-disjoint, so an image is never evaluated using a centroid estimated from that same image.

For transformers, semantic evidence is distributed across several transformer blocks rather than concentrated in a single feature space. For each feasible victim-target pair $( v , t )$ , we therefore use $L$ selected blocks and compute the cosine similarity of each internal feature $z _ { \ell } ( x )$ to the corresponding victim and target reference centroids $\mu _ { \ell , v }$ and $\mu _ { \ell , t }$ , obtaining $c _ { \ell , v } ( x )$ and $c _ { \ell , t } ( \boldsymbol { x } )$ The 2L similarities are concatenated into $\phi _ { v , t } ( x )$ and standardized before a regularized joint logistic readout, $r _ { v , t } ( \boldsymbol { x } ) \ = \ \sigma ( \boldsymbol { w } ^ { \top } \widetilde { \phi } _ { v , t } ( \boldsymbol { x } ) + b )$ Here, $r _ { v , t } ( x )$ measures target compatibility, while $p _ { v } ^ { \mathrm { i n t } } ( x ) = 1 - r _ { v , t } ( x )$ represents internal victim support. The readout is trained only from victim/target pseudo-labels produced by the frozen segmentation head on the reference population, i.e., no semantic or attack labels are used. The joint readout provides the direct victim-evidence signal, while the layer-wise similarities are retained as semantic coordinates for the residual geometry.

Attack-unaware pair discovery. Every feasible ordered pair $( v , t )$ is initially treated as a candi date victim-target relation. For CNNs, the primary signal is internal-to-head contradiction: evidence for $( v , t )$ increases when pixels retain victim-class evidence in the internal representation while the segmentation head predicts t. This evidence is aggregated over frozen, image-disjoint reference views and retained only when the support is sufficiently stable and separated from competing pairs. The resulting pair defines the victim-to-target direction used by the subsequent localization and repair stages. Thus, RePAIR can identify the attack direction directly from model-derived evidence, without requiring prior knowledge of the victim or target class.

For transformers, candidate structure is first identified using lightweight head, transfer, centroid, and structural cues. Supported pairs are then evaluated using image-disjoint joint-LogReg out-offold (OOF) evidence, with a separate OOF state for each pair. Stable family proposals are formed from structural, mechanism, and family-verification evidence. The frozen rank prior is retained whenever it passes the candidate viability test. If the prior family is not viable, another family may be selected only through the conservative rescue path, which requires bilateral leave-one-out support and bilateral structural consistency.

After the family has been frozen, its two possible victim-target directions are evaluated using FullEdge, FamilyVerify, and eligible Mechanism evidence. A direction is accepted only when the required direction agreement is unique. Structural evidence is not a regular direction vote and is used only to corroborate the rescue rule when the Mechanism direction channel is ineligible. If family viability or direction evidence is insufficient, the mapping remains inactive. The exact release contract is given in Appendix A.1.6.

Safety-approved reference state. After a victim-target pair has been selected, a correct mapping is not yet sufficient because the unlabeled image pool may itself contain poisoned samples. The builder therefore screens candidate reference images using image-disjoint cross-fitting and a victim-safety firewall. Images with unstable internal predictions are first rejected, followed by a safety filter that removes images showing severe victim-class collapse or moderate victim degradation accompanied by an unusually strong victim-to-target contradiction. These filters are applied sequentially and monotonically: rejected images are not reintroduced to satisfy a desired reference-set size, and thresholds are not relaxed to rescue an insufficient population.

The builder then checks whether the remaining images provide sufficient reliable support and semantic coverage for the selected victim-target pair. If not, the repair state remains inactive and no repair is applied at inference time. Otherwise, all runtime components, including semantic references, probability statistics, feature-space reference banks, and repair calibrations, are constructed exclusively from the final safety-approved population. We use this term rather than clean reference set because the defense has no access to the true provenance of the images and therefore cannot guarantee that they are trigger-free.

The resulting state provides the empirical victim-side reference used by the runtime defense. It is not treated as the unknown clean distribution $p ^ { \star }$ and is not assumed to be uncontaminated, i.e., it is instead a model-derived reference population that has passed the defense’s one-way safety checks.

Pair-consistent localization. Localization combines direct semantic contradiction with residual tests for pixels whose internal victim evidence is weak or distributed across multiple features. A shared probability branch compares the observed target-over-victim probability excess $d _ { p } ( x ) =$ $q _ { t } ( x ) - q _ { v } ( x )$ with a semantically matched reference excess. Victim and target reference neighborhoods provide matched reference values and a semantic interpolation coordinate, yielding the expected excess $\widehat { d } _ { \mathrm { r e f } } ( x )$ . RePAIR measures the one-sided residual $D _ { \mathrm { p r o b } } ( x ) = [ d _ { p } ( x ) - \widehat { d } _ { \mathrm { r e f } } ( x ) ] _ { + } ,$ which is calibrated image-disjointly on the frozen safety-approved population using an empirical reference-pixel budget. This budget controls the probability-residual calibration and is distinct from the repair margin. Matching is performed after removing the victim-target semantic axis, so the expected margin is conditioned on the remaining semantic geometry rather than on the displacement being tested.

CNNs combine the preserved semantic contradiction with target-manifold, self-counterfactual, and probability-residual cues. Transformer use five sequential, mutually exclusive stages restricted to baseline head-target pixels: semantic contradiction, residual victim-manifold rupture, residual target-manifold rupture, residual counterfactual manifold trajectory, and residual semantic probability. Their union defines $M _ { \mathrm { T F } }$ and satisfies $M _ { \mathrm { T F } } \subseteq \{ \widehat { Y } ^ { \mathrm { h e a d } } = t \}$ The staged construction records why a pixel was selected, prevents later branches from relabeling an earlier decision, and enables branch-specific repair. Exact masks and image-disjoint calibration rules are given in Appendix A.2.4.

Feature-first repair. RePAIR first repairs the internal representation of localized pixels before applying any residual output correction. For each localized feature, the discovered victim-target direction is removed from the retrieval geometry, and safety-approved victim references are searched in the orthogonal complement. The retrieved neighborhood is aggregated and the feature is transported toward the victim reference with branch-dependent strength, while preserving its original norm. The modified representation is then passed through the unchanged remainder of the network. By matching in the orthogonal complement, the reference search is not dominated by the same victim-target displacement that the repair is intended to reverse.

If the feature repair does not fully restore the victim prediction, RePAIR applies a conservative output correction restricted to the discovered victim-target pair. The correction is calibrated from safety-approved victim references and can be partial depending on the repair branch. Thus, feature transport and output correction address the same victim-target discrepancy at different stages: feature transport moves the internal representation toward victim semantics, while the output step corrects residual target predictions by directly modifying only the victim and target logits.

The resulting feature transport approximates the reverse-transport direction without requiring the unknown clean distribution or transport magnitude. The discovered pair provides the direction, while the safety-approved victim references provide the local repair target.

Output safety and decision isolation. Let $\widehat { Y }$ <sup>rep</sup> denote the prediction after repair but before output safety. The No New Target Veto restores Yb <sup>head</sup> whenever the repair changes a pixel outside the authorized mask to the discovered target class t, provided that the original head prediction was not t. Thus, the veto prevents unsupported new target predictions outside the repair support while leaving existing target predictions unaffected. The veto is intentionally asymmetric and only constrains the creation of the discovered target.

## 5 EXPERIMENTS

## 5.1 EXPERIMENTAL SETTING

We evaluate RePAIR on Cityscapes (Cordts et al., 2016), PASCAL VOC 2012 (Everingham et al., 2015) and ADE20K (et al., 2017). We consider two representative segmentation architectures, DeepLabV3+ with a ResNet-50 backbone (Chen et al., 2018) and SegFormer with a MiT-B5 backbone (Xie et al., 2021).

We evaluate backdoor attacks at a 10% poison rate, including IBA with NNI and PRL strategies, ConSeg, OFBA, and BadSeg with all four attack configurations. For all object-to-background attacks (IBA NNI, IBA PRL, ConSeg, OFBA, and BadSeg-O2B), we use car→road as the victimtarget pair. For the remaining attack configurations, we use car→person for object-to-object, sidewalk→road for background-to-background, and road→car for background-to-object attacks.

We compare RePAIR against all defense methods, i.e., pruning, fine-tuning, Segment Cleanse, and ABL. Performance is evaluated using Attack Success Rate (ASR), benign mean intersection-overunion (mIOU-B), and attacked mIoU (mIOU-A). For BadSeg-B2O, ASR is restricted to groundtruth victim pixels inside the designated attack region. More details on metric definitions and experimental configurations are provided in Appendix A.3.

## 5.2 NUMERICAL RESULTS

Attack success rate. Table 1 reports attack success rate before and after applying the evaluated defenses for the Cityscapes dataset. Lower values indicate stronger attack mitigation. ASR measures the fraction of victim-class pixels that are misclassified as the attacker-selected target class on triggered inputs.

For DeepLabV3+, the undefended model achieves high ASR across all attacks, ranging from 60.83% to 97.41%. Fine-tuning and pruning provide little mitigation and leave the ASR high for most attacks. In contrast, RePAIR consistently reduces the ASR to below 7.86% across all evaluated attacks. This consistent performance across different attack types is not observed for the other defense methods. Segment Cleanse achieves effective mitigation for IBA and OFBA, but performs substantially worse on the BadSeg attacks compared to our defense. Overall, RePAIR achieves the best or second-best ASR for most attacks and maintains strong mitigation consistently across the full attack set.

For SegFormer, the undefended model shows lower ASR for several attacks, with values ranging from 26.43% to 95.46%. Fine-tuning and pruning again provide limited mitigation, while Segment Cleanse also fails to consistently reduce the ASR. ABL provides more consistent mitigation across the evaluated attacks. RePAIR achieves lower ASR than all other evaluated defenses on all backdoor attacks, reducing the ASR to below 0.54% and frequently to zero. Thus, for these cases, our defense prevents the attack from successfully redirecting any victim-class pixels to the target class.

Table 1: Cityscapes results across backdoor attacks. ASR is the primary metric, with lower values indicating better defense performance. mIoU-A and mIoU-B are averaged across the eight attacks and reported with their minimum and maximum deviations from the mean.
<table><tr><td>Model</td><td>Method</td><td>IBA (NNI)</td><td>IBA (PRL)</td><td>ConSeg</td><td>OFBA</td><td>BadSeg-O2B</td><td>BadSeg-O2O</td><td>BadSeg-B2O</td><td>BadSeg-B2B</td><td>mIoU-A</td><td>mIoU-B</td></tr><tr><td rowspan="6">Deep- LabV3+</td><td>no defense</td><td>88.56</td><td>97.41</td><td>86.23</td><td>60.83</td><td>95.49</td><td>97.03</td><td>96.87</td><td>96.82</td><td>71.14+6.498</td><td>73.28+6.00 -10.01</td></tr><tr><td>fine-tuning</td><td>56.36</td><td>12.35</td><td>84.54</td><td>6.31</td><td>13.42</td><td>81.91</td><td>86.36</td><td>97.69</td><td>35.86+14.003</td><td>37.65+121.41 -21.43</td></tr><tr><td>pruning</td><td>89.60</td><td>98.13</td><td>90.07</td><td>71.36</td><td>96.63</td><td>96.55</td><td>97.58</td><td>97.25</td><td>70.13+6.82 -6.58</td><td>72.37+6.00 -9.91</td></tr><tr><td>Seg.-Cleanse</td><td>0.36</td><td>0.89</td><td>86.23</td><td>0.75</td><td>3.31</td><td>94.20</td><td>96.85</td><td>36.95</td><td>71.14+0.</td><td>74.14 +5.30 -7.31</td></tr><tr><td>ABL</td><td>83.94</td><td>94.55</td><td>1.12</td><td>0.78</td><td>17.29</td><td>90.36</td><td>86.19</td><td>92.17</td><td>36.66 +4.39 -6.48</td><td>40.97 +5.46 -4.08</td></tr><tr><td>RePAIR (ours)</td><td>1.18</td><td>1.01</td><td>1.51</td><td>1.10</td><td>2.35</td><td>1.00</td><td>0.46</td><td>7.86</td><td>73.83+3.97 -7.77</td><td>74.50+3.17 -5.69</td></tr><tr><td rowspan="6">Seg- Former</td><td>no defense</td><td>26.43</td><td>28.71</td><td>89.15</td><td>45.32</td><td>87.39</td><td>86.24</td><td>94.11</td><td>95.46</td><td>71.95+78</td><td>75.80+449 -27.52</td></tr><tr><td>fine-tuning</td><td>26.57</td><td>35.44</td><td>93.90</td><td>46.92</td><td>86.46</td><td>85.12</td><td>89.73</td><td>95.10</td><td>69.54±8.796</td><td>73.05 +5.87 -33.14</td></tr><tr><td>pruning</td><td>26.80</td><td>29.45</td><td>90.06</td><td>45.64</td><td>87.53</td><td>86.22</td><td>93.76</td><td>95.51</td><td>71.74+7.94 22.43</td><td>75.54+4.64 -28.54</td></tr><tr><td>Seg.-Cleanse</td><td>21.07</td><td>22.71</td><td>75.46</td><td>36.98</td><td>80.67</td><td>85.30</td><td>94.11</td><td>92.91</td><td>71.91+7.872 20.72</td><td>75.77 7+4.52 -26.49</td></tr><tr><td>ABL</td><td>10.89</td><td>15.09</td><td>0.71</td><td>9.11</td><td>62.48</td><td>2.32</td><td>73.35</td><td>17.60</td><td>20.87 7+9.14 -9.36</td><td>22.50 +8.80 -9.46</td></tr><tr><td>RePAIR (ours)</td><td>0.00</td><td>0.00</td><td>0.54</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.31</td><td>74.52+21784 -21.34</td><td>72.30+4.80 -19.16</td></tr></table>

Segmentation performance. Table 1 reports the mean mIoU-B and mIoU-A across all evaluated attacks in the two rightmost columns. The clean models, without a backdoor attack or defense, achieve an mIoU-B of 78.74% for DeepLabV3+ and 80.46% for SegFormer, representing the upper reference for benign segmentation performance. Under attack, the goal is to preserve this performance on benign classes while recovering the attacked classes.

The undefended backdoored models retain an mIoU-B of 73.28% for DeepLabV3+ and 75.80% for SegFormer, showing that the attacks primarily affect the targeted classes. In contrast, ABL substantially degrades mIoU-B, indicating that its mitigation affects benign predictions beyond the attacked classes. RePAIR maintains mIoU-B at 74.50% for DeepLabV3+ and 72.30% for SegFormer, while achieving the highest mIoU-A among the evaluated defenses. Thus, RePAIR combines recovery of the attacked classes with preservation of the model’s general segmentation performance.

Other datasets. We additionally evaluate RePAIR on PASCAL VOC 2012 and ADE20K, considering only Object-to-Background attack configurations for both datasets. On ADE20K, the evaluated attack configurations do not produce a sufficiently strong backdoor effect in the undefended models to support a meaningful mitigation study. This behavior is further analyzed theoretically in Appendix A.1, and the corresponding results are reported in Table 5.

PASCAL VOC differs from Cityscapes primarily in the number of semantic classes present within each image. Images in PASCAL VOC typically contain a single foreground object class together with a background label that aggregates all other classes. This different class composition affects the learned class representations and, consequently, the information available to the defense. Across the evaluated attacks, the undefended models achieve ASR ranging from 12.21% for NNI to 99.71% for ConSeg, while RePAIR reduces the remaining ASR to between 5.41% and 19.67%. The corresponding results are reported in Table 6 and 7.

## 6 CONCLUSION

We introduced RePAIR, a post-hoc, label-free, and attack-unaware defense that exploits surviving internal victim-class evidence to localize and selectively repair backdoor-affected pixels without retraining the compromised segmentation model. Across eight attack configurations on Cityscapes, RePAIR reduces the defended ASR to 0.46%-7.86% for DeepLabV3+ and to at most 0.54% for SegFormer, with complete suppression of the measured attack success in six of the eight SegFormer settings. At the same time, RePAIR preserves substantial segmentation utility, achieving an average mIoU-B of 74.50% for DeepLabV3+ and 72.30% for SegFormer, compared with 73.28% and 75.80% for the corresponding undefended models.

Beyond the overall mitigation performance, the experiments show that effective backdoor repair can be achieved without explicitly reconstructing the trigger or modifying the compromised model. The directed-transport formulation provides a view of the attack as a structured victim-to-target shift, while the internal representation provides a model-derived reference for reversing this shift selectively. Across convolutional and transformer-based architectures, these results indicate that surviving internal semantic evidence can serve as a basis for attack-unaware localization and constrained pixel-wise repair.

## AI USE STATEMENT

In this work, we used generative AI tools to assist coding, debugging, mathematical-notation consistency, and LaTeX drafting. We did not use generative AI tools to generate experimental measurements or to replace human evaluation of the theoretical claims. AI-assisted outputs were reviewed for mathematical consistency, and generated LaTeX was compiled and visually inspected. We take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## ETHICS STATEMENT

This work studies defenses against data-poisoning backdoors in semantic segmentation, with emphasis on safety-critical perception. Published attacks are used only as controlled evaluation mechanisms for the proposed defense. The paper does not propose a new attack or expand the attacker’s capabilities. Any release of code, models, or configuration files will follow the licenses of the underlying datasets and software and will be framed for defensive research and reproducible evaluation. The study does not involve human-subject experiments.

## REPRODUCIBILITY STATEMENT

We provide the implementation of RePAIR, including the model-specific feature extraction, victimtarget pair selection, reference construction, localization, feature transport, and output-safety procedures. We use the publicly available Cityscapes, PASCAL VOC 2012, and ADE20K datasets and evaluate the method on DeepLabV3+ with ResNet-50 and SegFormer with MiT-B5. The attack and defense configurations, evaluation metrics, and implementation details are described in the main paper and the appendix.

## REFERENCES

Bilal Hussain Abbasi, Zirui Gong, Yanjun Zhang, Shang Gao, Antonio Robles-Kelly, and Leo Zhang. Conseg: Contextual backdoor attack against semantic segmentation, 2025. URL https://arxiv.org/abs/2507.19905.

N. Ahmed, T. Natarajan, and K. R. Rao. Discrete cosine transform. IEEE Transactions on Computers, C-23(1):90–93, 1974. doi: 10.1109/T-C.1974.223784.

Senay Cakir, Marcel Gauß, Kai Happeler, Yassine Ounajjar, Fabian Heinle, and Reiner Marchthaler.¨ Semantic segmentation for autonomous driving: Model evaluation, dataset generation, perspective comparison, and real-time capability, 2022. URL https://arxiv.org/abs/2207. 12939.

Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam. Encoderdecoder with atrous separable convolution for semantic image segmentation. In Vittorio Ferrari, Martial Hebert, Cristian Sminchisescu, and Yair Weiss (eds.), Computer Vision – ECCV 2018, pp. 833–851, Cham, 2018. Springer International Publishing. ISBN 978-3-030-01234-2.

Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

Bolei Zhou et al. Scene parsing through ade20k dataset. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017.

Mark Everingham, S. M. Ali Eslami, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserman. The pascal visual object classes challenge: A retrospective. International Journal ofComputer Vision, 111(1):98–136, 2015.

Yansong Gao, Chang Xu, Derui Wang, Shiping Chen, Damith C. Ranasinghe, and Surya Nepal. Strip: A defence against trojan attacks on deep neural networks. In Proceedings of the 35th Annual Computer Security Applications Conference, ACSAC ’19, pp. 113–125, New York, NY, USA, 2019. Association for Computing Machinery. ISBN 9781450376280. doi: 10.1145/3359789. 3359790. URL https://doi.org.

Tianyu Gu, Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. Badnets: Evaluating backdooring attacks on deep neural networks. IEEE Access, 7:47230–47244, 2019. doi: 10.1109/ACCESS. 2019.2909068.

Haoheng Lan, Jindong Gu, Philip Torr, and Hengshuang Zhao. Influencer backdoor attack on semantic segmentation. In The Twelfth International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ 01977a08d8e76cf4df40b5b57407c9eb-Abstract-Conference.html.

Yige Li, Xilie Xu, Ning Yu, Teng Li, and Ji Liu. Anti-backdoor learning: Training clean models on poisoned data. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 14930–14941. Curran Associates, Inc., 2021a.

Yiming Li, Yanjie Li, Yalei Lv, Yong Jiang, and Shu-Tao Xia. Hidden backdoor attack against semantic segmentation models. In ICLR Workshop on Security and Safety in Machine Learning Systems, 2021b. URL https://arxiv.org/abs/2103.04038.

Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. Fine-pruning: Defending against backdooring attacks on deep neural networks. In Research in Attacks, Intrusions, and Defenses (RAID), pp. 273–294. Springer, 2018.

Xiaogeng Liu, Minghui Li, Haoyu Wang, Shengshan Hu, Dengpan Ye, Hai Jin, Libing Wu, and Chaowei Xiao. Detecting backdoors during the inference stage based on corruption robustness consistency. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20960–20969, June 2023.

Wanlun Ma, Derui Wang, Ruoxi Sun, Minhui Xue, Sheng Wen, and Yang Xiang. The ”beatrix” resurrections: Robust backdoor detection via gram matrices. 01 2023. doi: 10.14722/ndss.2023. 23069.

Yunyao Mao, Ning Wang, Wengang Zhou, and Houqiang Li. Object-free backdoor attack and defense on semantic segmentation. Computers & Security, 132:103365, 2023.

Zeyang Sha, Xinlei He, Pascal Berrang, Mathias Humbert, and Yang Zhang. Fine-tuning is all you need to mitigate backdoor attacks, 12 2022.

Saeid Asgari Taghanaki, Kumar Abhishek, Joseph Paul Cohen, Julien Cohen-Adad, and Ghassan Hamarneh. Deep semantic segmentation of natural and medical images: A review, 2024. URL https://arxiv.org/abs/1910.07655.

Bolun Wang, Yuanshun Yao, Shawn Shan, Huiying Li, Bimal Viswanath, Haitao Zheng, and Ben Y. Zhao. Neural cleanse: Identifying and mitigating backdoor attacks in neural networks. In Proceedings of the IEEE Symposium on Security and Privacy (SP), pp. 707–723, 2019. doi: 10.1109/SP.2019.00031.

Enze Xie, Wenhai Wang, Zhiding Yu, Anima Anandkumar, Jose M. Alvarez, and Ping Luo. Segformer: Simple and efficient design for semantic segmentation with transformers. In M. Ranzato, A. Beygelzimer, Y. Dauphin, P.S. Liang, and J. Wortman Vaughan (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 12077–12090. Curran Associates, Inc., 2021. URL https://proceedings.neurips.cc/paper\_files/ paper/2021/file/64f1f27bf1b4ec22924fd0acb550c235-Paper.pdf.

Yi Zeng, Won Park, Z. Morley Mao, and Ruoxi Jia. Rethinking the backdoor attacks’ triggers: A frequency perspective. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 16473–16481, October 2021.

Guangsheng Zhang, Huan Tian, Leo Zhang, Tianqing Zhu, Ming Ding, Wanlei Zhou, and Bo Liu. Poisoning the pixels: Revisiting backdoor attacks on semantic segmentation, 2026. URL https: //arxiv.org/abs/2603.16405.

## A APPENDIX

## A.1 THEORY: FULL STATEMENTS AND PROOFS

In the following, we formalize the local transport operator, distinguish its inverse from the transposed transport coupling, and describe its operational interpretation for RePAIR. The emphasis here is on the formal domain, feasibility conditions, and proof obligations underlying the theoretical claims.

## A.1.1 PROBABILITY SPACE AND STANDING CONVENTIONS

Let Ω denote the evaluated pixel population and

$$
Y : \Omega  [ K ] \cup \{ \varpi \} \ , \qquad \widehat { Y } ^ { \mathrm { h e a d } } , \widehat { Y } ^ { \mathrm { r e p } } , \widehat { Y } ^ { \mathrm { d e f } } : \Omega  [ K ] \ ,
$$

where K is the ground-truth class, $\widehat { Y } ^ { \mathrm { h e a d } }$ the undefended head prediction, $\widehat { Y } ^ { \mathrm { r e p } }$ the repaired prediction before output safety, and $\widehat { Y } ^ { \mathrm { d e f } }$ the final defended prediction. Pixels with $Y = \varnothing$ are excluded from semantic metrics.

Let $T$ denote the event that a trigger is present, and assume $\mathbb { P } ( T ) > 0$ and $\mathbb { P } ( T ^ { c } ) > 0$ . Define

$$
\mathbb { P } _ { \mathrm { t r i g } } ( \cdot ) = \mathbb { P } ( \cdot \mid T )\tag{9}
$$

as the conditional distribution over triggered inputs, and

$$
\mathbb { P } _ { \mathrm { c l } } ( \cdot ) = \mathbb { P } ( \cdot \mid T ^ { c } )\tag{10}
$$

as the conditional distribution over clean inputs.

For victim class v and target class $t \neq v ,$ , assume $\mathbb { P } _ { \mathrm { t r i g } } ( Y = v ) > 0$ and define

$$
\mathbb { P } _ { v } ( B ) : = \mathbb { P } _ { \mathrm { t r i g } } ( B \mid Y = v )\tag{11}
$$

as the conditional probability over triggered pixels whose ground-truth class is v. Let

$$
A : = \{ \widehat { Y } ^ { \mathrm { h e a d } } = t \}\tag{12}
$$

denote the event that the undefended model predicts the target class. We then define the baseline and defended attack success rates as

$$
\mathrm { A S R } _ { \mathrm { b a s e } } = \mathbb { P } _ { v } ( A ) , \qquad \mathrm { A S R } _ { \mathrm { d e f } } = \mathbb { P } _ { v } ( \widehat { Y } ^ { \mathrm { d e f } } = t ) .\tag{13}
$$

For theoretical evaluation, $( v , t )$ denotes the true attacked victim-target pair, which unknown to the defense. The attack-unaware builder either releases an active runtime state with discovered pair $\widehat { P } = ( \widehat { v } , \widehat { t } )$ or abstains and releases an inactive fail-closed state. Let $R \in \{ 0 , 1 \}$ indicate whether an active state is released. We use the following conditions throughout the analysis:

$$
C _ { \mathrm { p a i r } } : \quad R = 1 \mathrm { a n d } ( \hat { v } , \hat { t } ) = ( v , t ) ,\tag{14}
$$

$$
C _ { \mathrm { t g t } } : ~ R = 1 \mathrm { a n d } \hat { t } = t \ .\tag{15}
$$

Clearly, $C _ { \mathrm { p a i r } }$ implies $C _ { \mathrm { t g t } }$ . All pair-conditioned runtime quantities are constructed from the released state and therefore use $( \hat { v } , \hat { t } )$ rather than the unknown pair $( v , t )$ . The residual-ASR decomposition below applies to the actually executed defense state without assuming $C _ { \mathrm { p a i r } }$ . In contrast, statements that relate localization error to internal victim evidence for the true pair require $C _ { \mathrm { p a i r } } .$ Guarantees concerning the creation of the true target class require only $C _ { \mathrm { t g t } }$ . If $R = 0$ , fail-closed inference leaves the undefended prediction unchanged.

Let $\rho _ { v } ~ \in ~ \{ 0 , 1 \}$ denote the evaluation-side internal victim-evidence event for the true pair. For CNNs, $\rho _ { v } = { \bf 1 } \{ \widehat { Y } ^ { \mathrm { p r o t } } = v \}$ . For transformers, $\rho _ { v } = 1 \{ p _ { v } ^ { \mathrm { i n t } } > \tau _ { \mathrm { s e m } } \}$ , where $\tau _ { \mathrm { s e m } }$ is the frozen, pair-specific clean-null threshold for internal victim support. In the official transformer state it is calibrated as the 0.995 empirical quantile of image-disjoint out-of-fold victim-support scores on the final safety-approved reference population. For an active released state, the implemented direct semantic localization arm uses the discovered pair,

$$
M _ { \mathrm { s e m a n t i c } } = { \bf 1 } \{ \widehat { Y } ^ { \mathrm { h e a d } } = \widehat { t } \} \rho _ { \widehat { v } } .\tag{16}
$$

Here $\rho _ { \hat { v } }$ is constructed by the same architecture-specific rule using the released pair. Let $M : \Omega $ $\{ 0 , 1 \}$ denote the final localization indicator. The implemented masks preserve the semantic arm additively, so

$$
M _ { \mathrm { s e m a n t i c } } \leq M .\tag{17}
$$

Under $C _ { \mathrm { p a i r } }$ , the runtime quantities satisfy $\rho _ { \hat { v } } = \rho _ { v }$ and $M _ { \mathrm { s e m a n t i c } } = \mathbf { 1 } \{ \widehat { Y } ^ { \mathrm { h e a d } } = t \} \rho _ { v }$ . If $R = 0 ,$ we set $\hat { M } _ { \mathrm { s e m a n t i c } } = M = 0$ consistently with fail-closed inference.

To make all statements well-defined at zero prevalence, we define

$$
\eta _ { \mathrm { r e f } } = { \left\{ \begin{array} { l l } { \mathbb { P } _ { v } ( \rho _ { v } = 0 \mid A ) , } & { \mathbb { P } _ { v } ( A ) > 0 } \\ { 0 , } & { \mathbb { P } _ { v } ( A ) = 0 } \end{array} \right. } ,\tag{18}
$$

$$
\eta _ { \mathrm { l o c } } = { \left\{ \mathbb { P } _ { v } ( M = 0 \mid A ) , \mathbb { P } _ { v } ( A ) > 0 \ , \right. }\tag{19}
$$

and

$$
\varepsilon _ { \mathrm { r e p } } = \left\{ \begin{array} { l l } { \mathbb { P } _ { v } ( \widehat { Y } ^ { \mathrm { d e f } } = t \mid M = 1 , A ) , } & { \mathbb { P } _ { v } ( M = 1 , A ) > 0 } \\ { 0 , } & { \mathbb { P } _ { v } ( M = 1 , A ) = 0 } \end{array} \right. ,\tag{20}
$$

Finally,

$$
\delta _ { \mathrm { n e w } } = \mathbb { P } _ { v } ( \widehat { Y } ^ { \mathrm { d e f } } = t , A ^ { c } ) \ .\tag{21}
$$

These zero-prevalence conventions change no positive-mass conditional and make all products in the subsequent bounds unambiguous.

## A.1.2 FIXED-SIMPLEX GEOMETRY

The class-probability simplex and its tangent space are

$$
\Delta ^ { K - 1 } = \{ p \in \mathbb { R } _ { \geq 0 } ^ { K } : \mathbf { 1 } ^ { \top } p = 1 \} \ ,\tag{22}
$$

$$
\mathcal { T } _ { \Delta } = \left\{ d \in \mathbb { R } ^ { K } : \mathbf { 1 } ^ { \top } d = 0 \right\} \ .\tag{23}
$$

For pair $( v , t )$ define

$$
d _ { v t } = e _ { t } - e _ { v } \in \mathcal { T } _ { \Delta } \mathrm { ~ , ~ }\tag{24}
$$

where $e _ { c }$ denotes the one-hot basis vector for class c.

Lemma 1 (Directed victim–target displacement). Let $p ^ { \star } \in \Delta ^ { K - 1 }$ and let

$$
q = p ^ { \star } + \xi d _ { v t } , 0 \leq \xi \leq p _ { v } ^ { \star } ,\tag{25}
$$

with $q ~ \in ~ \Delta ^ { K - 1 }$ . Here, $\xi$ denotes the amount of probability mass transported from the victim coordinate v to the target coordinate $t ,$ and $\boldsymbol { d _ { v t } } = \boldsymbol { e _ { t } } - \boldsymbol { e _ { v } }$ is the corresponding directed victimto-target displacement. The constraint $0 ~ \le ~ \xi ~ \le ~ p _ { v } ^ { \star }$ ensures that the victim coordinate remains nonnegative. For $\psi _ { v t } ( p ) = p _ { v } - p _ { t }$

$$
\psi _ { v t } ( q ) = \psi _ { v t } ( p ^ { \star } ) - 2 \xi .\tag{26}
$$

Moreover, with the sign convention positive on the victim side, the signed Euclidean distance to $\mathcal { H } _ { v t } = \{ p : p _ { v } = p _ { t } \}$ is

$$
\mathrm { s d i s t } ( p , \mathcal { H } _ { v t } ) = \frac { p _ { v } - p _ { t } } { \sqrt { 2 } } \mathrm { ~ . ~ }\tag{27}
$$

Proof. Equation 25 gives $q _ { v } = p _ { v } ^ { \star } - \xi$ and $q _ { t } = p _ { t } ^ { \star } + \xi ,$ , proving the margin identity. The hyperplane normal is $e _ { v } - e _ { t } ,$ whose Euclidean norm is $\sqrt { 2 }$ . Thus, the stated signed-distance formula follows.

Corollary 2 (Pairwise preference flip). $I f p _ { v } ^ { \star } > p _ { t } ^ { \star }$ , then the strict pairwise ordering $q _ { t } > q _ { v }$ holds whenever

$$
\xi > \frac { p _ { v } ^ { \star } - p _ { t } ^ { \star } } { 2 } \ .\tag{28}
$$

At equality the pair is tied.

The corollary concerns only the victim-target pairwise boundary. A complete change of arg max additionally requires $q _ { t }$ to exceed every other class coordinate. The construction is a local analytical model, i.e., RePAIR does not observe $p ^ { \star }$ at runtime.

## A.1.3 FORMAL TRANSPORT OPERATOR AND COUPLING

Section 3 introduces the local operator $T _ { \xi } ( p ) = p + \xi ( e _ { t } - e _ { v } )$ . Here we record its feasible inverse domain and the coupling construction used in the proofs.

Define

$$
\mathcal { D } _ { \xi } = \left\{ p \in \Delta ^ { K - 1 } : p _ { v } \geq \xi \right\} \ , \qquad \mathcal { R } _ { \xi } = \left\{ q \in \Delta ^ { K - 1 } : q _ { t } \geq \xi \right\} \ .\tag{29}
$$

Lemma 3 (Local inverse operator). For fixed $\xi \geq 0 ,$ the map

$$
T _ { \xi } : \mathcal { D } _ { \xi } \to \mathcal { R } _ { \xi } , \qquad T _ { \xi } ( p ) = p + \xi ( e _ { t } - e _ { v } ) ,\tag{30}
$$

is bijective with

$$
T _ { \xi } ^ { - 1 } ( q ) = q + \xi ( e _ { v } - e _ { t } ) .\tag{31}
$$

Proof. For $p \in { \mathcal { D } } _ { \xi }$ , the victim coordinate remains nonnegative after subtracting $\xi ,$ while all coordinates still sum to one, so $T _ { \xi } ( p ) \in \Delta ^ { K - 1 }$ and its target coordinate is at least $\xi .$ Conversely, for $q \in \mathcal { R } _ { \xi }$ , subtracting ξ from $q _ { t }$ and adding it to $q _ { v }$ yields a valid simplex point in $\mathcal { D } _ { \xi }$ . Direct substitution shows that the two maps are mutual inverses. □

Lemma 4 (One-edge coupling). Let $p \in { \mathcal { D } } _ { \xi }$ and $q = T _ { \xi } ( p )$ . Define $\gamma _ { \xi } \in \mathbb { R } _ { \ge 0 } ^ { K \times K }$ by

$$
( \gamma _ { \xi } ) _ { v v } = p _ { v } - \xi \ , \qquad ( \gamma _ { \xi } ) _ { v t } = \xi \ , \qquad ( \gamma _ { \xi } ) _ { k k } = p _ { k } \quad ( k \ne v ) \ ,
$$

with all remaining entries zero. Then

$$
\gamma _ { \xi } { \bf 1 } = p \ , \qquad \gamma _ { \xi } ^ { \top } { \bf 1 } = q \ .\tag{32}
$$

Hence $\gamma _ { \xi }$ is a feasible coupling between the forward marginals p and $q ,$ while $\gamma _ { \xi } ^ { \top }$ is a feasible coupling from q back to p.

Proof. The victim row sums to $( p _ { v } - \xi ) + \xi = p _ { v }$ and every other row to $p _ { k }$ . The victim column sums to $p _ { v } - \xi = q _ { v } .$ , the target column to $p _ { t } + \xi = q _ { t }$ , and every other column to $p _ { k } \ = \ q _ { k } .$ Transposition exchanges these marginals. □

The last statement concerns a joint measure, not a function on probability vectors. Thus $\gamma _ { \xi } ^ { \top }$ represents reversed mass flow but is not the operator $T _ { \xi } ^ { - 1 }$ . No optimality claim is made for either coupling without specifying a transport cost.

## A.1.4 INTERNAL-REFERENCE SURVIVAL AND LOCALIZATION

Lemma 5 (Surviving victim evidence). Under the conventions in Section A.1.1,

$$
\mathbb { P } _ { v } ( A , \rho _ { v } = 1 ) = ( 1 - \eta _ { \mathrm { r e f } } ) \mathrm { A S R _ { b a s e } } \ .\tag{33}
$$

Proof. $\operatorname { I f } \mathbb { P } _ { v } ( A ) > 0$ , the multiplication rule gives $\mathbb { P } _ { v } ( A , \rho _ { v } = 1 ) = \mathbb { P } _ { v } ( \rho _ { v } = 1 \mid A ) \mathbb { P } _ { v } ( A )$ , which is the claimed expression. $\operatorname { I f } \mathbb { P } _ { v } ( A ) = 0$ , both sides are zero by definition. □

The relevant condition is therefore not complete encoder invariance. $\eta _ { \mathrm { r e f } }$ measures only the fraction of successful baseline attacks for which the direct internal victim-evidence event is absent.

Proposition 6 (Internal evidence bounds localization misses). Assume $C _ { \mathrm { p a i r } }$ and Equation 17. Then

$$
\eta _ { \mathrm { l o c } } \leq \eta _ { \mathrm { r e f } } .\tag{34}
$$

Proof. Under $C _ { \mathrm { p a i r } }$ , the released pair satisfies $( \hat { v } , \hat { t } ) \ = \ ( v , t )$ , so the implemented semantic arm is exactly $M _ { \mathrm { s e m a n t i c } } = \mathbf { 1 } \{ \widehat { Y } ^ { \mathrm { h e a d } } = t \} \rho _ { v } . \ \mathrm { I f ~ } \mathbb { P } _ { v } ( A ) = 0$ , both quantities are zero by convention. Otherwise, on $A , \rho _ { v } = 1$ implies $M _ { \mathrm { s e m a n t i c } } = 1$ , and Equation 17 implies $M = \mathrm { { 1 } }$ . Therefore $\{ M = 0 \} \cap A \subseteq { \dot { \{ \rho _ { v } = 0 \} \cap { \bar { A } } } }$ . Conditioning on A proves the inequality. □

Additional residual branches may reduce localization misses beyond the direct semantic arm but cannot invalidate this inclusion.

## A.1.5 DIRECTED PAIR EVIDENCE AND POPULATION SEPARATION

Pair discovery is architecture specific, so we first state a generic scalar separation result. For each ordered pair $( a , b )$ , let $G _ { i } ( a , b )$ be an attack-unaware image-level evidence statistic and

$$
g _ { a b } : = \mathbb { E } [ G _ { i } ( a , b ) ] \mathrm { ~ . ~ }\tag{35}
$$

Given n builder images for which the statistic is defined, let

$$
\widehat { g } _ { a b } : = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } G _ { i } ( a , b )\tag{36}
$$

denote the corresponding empirical statistic. For analysis only, let $\nu _ { a b }$ denote the expected value of the same statistic under a corresponding no-backdoor reference population and define

$$
\beta _ { a b } : = g _ { a b } - \nu _ { a b } , g _ { a b } = \nu _ { a b } + \beta _ { a b } .\tag{37}
$$

Neither $\nu _ { a b }$ nor $\beta _ { a b }$ is observed by the defense.

For CNNs, one primitive statistic is

$$
G _ { i } ^ { \mathrm { C N N } } ( a , b ) = \frac { 1 } { | \Omega _ { i } | } \sum _ { x \in \Omega _ { i } } \mathbf { 1 } \{ \widehat { Y } ^ { \mathrm { p r o t } } ( x ) = a \land \widehat { Y } ^ { \mathrm { h e a d } } ( x ) = b \} \ .\tag{38}
$$

For transformers, one primitive OOF statistic is a fixed upper-quantile summary of internal victim support on head-target pixels,

$$
G _ { i } ^ { \mathrm { T F } } ( a , b ) = Q _ { \tau } \left( p _ { a } ^ { \mathrm { i n t } } ( x ) : \widehat { Y } ^ { \mathrm { h e a d } } ( x ) = b \right) ,\tag{39}
$$

when the eligible set is nonempty. Pairs for which this primitive statistic is undefined are outside the scope of the scalar proposition below. The full implementation also uses support, structural, and held-out mechanism evidence.

Proposition 7 (Sufficient scalar pair separation). Suppose $( v , t )$ is the true attacked pair and, for every competitor $( a , b ) \neq ( v , t )$

$$
| ( \widehat { g } _ { v t } - \widehat { g } _ { a b } ) - ( g _ { v t } - g _ { a b } ) | \leq \delta _ { \mathrm { e s t } } .\tag{40}
$$

If

$$
\beta _ { v t } - \operatorname* { m a x } _ { ( a , b ) \neq ( v , t ) } \beta _ { a b } > \operatorname* { m a x } _ { ( a , b ) \neq ( v , t ) } ( \nu _ { a b } - \nu _ { v t } ) + \delta _ { \mathrm { e s t } } ,\tag{41}
$$

then $\widehat { g } _ { v t } > \widehat { g } _ { a b }$ for every competitor.

Proof. For any competitor,

$$
g _ { v t } - g _ { a b } = \left( \beta _ { v t } - \beta _ { a b } \right) - \left( \nu _ { a b } - \nu _ { v t } \right) \ .
$$

The stated condition implies $g _ { v t } - g _ { a b } > \delta _ { \mathrm { e s t } }$ for every competitor. Equation 40 then gives $\widehat { g } _ { v t } -$ $\widehat { g } _ { a b } > 0$ □

The result is sufficient, not necessary, and is conditional on the stated uniform score-difference error bound. It should not be read as a finite-sample guarantee for the complete transformer mapping selector, whose family and direction decisions use multiple evidence views rather than a single scalar pair ranking.

## A.1.6 TRANSFORMER MAPPING RELEASE CONTRACT

Transformer mapping is modeled as a partial release rule over attack-unaware builder evidence. Let $\mathcal { F }$ denote the set of feasible undirected class families. Pair-owned out-of-fold evidence provides structural, mechanism, and family verification views for each candidate family. These views define a stable rank prior together with a proposal set

$$
\mathcal { F } _ { \mathrm { p r o p } } = \mathcal { C } _ { S } \cup \mathcal { C } _ { M } \cup \mathcal { C } _ { V } \enspace ,\tag{42}
$$

where ${ \mathcal { C } } _ { S } , { \mathcal { C } } _ { M }$ , and $\mathcal { C } _ { V }$ are the stable candidate cores induced by the corresponding evidence views.

$\mathrm { I f } \mathcal { F } _ { \mathrm { p r o p } } = \emptyset$ , the builder releases no mapping and returns $( R , { \widehat { P } } ) = ( 0 , \bot )$ . Otherwise, let $F _ { \mathrm { p r i o r } } \in$ $\mathcal { F } _ { \mathrm { p r o p } }$ denote the leading family under the frozen stable rank prior. Let

$$
\mathsf { V } _ { \mathrm { b a s e } } ( F ) \in \{ 0 , 1 \}\tag{43}
$$

denote the frozen candidate viability test. The builder preserves $F _ { \mathrm { p r i o r } }$ whenever $\mathsf { V } _ { \mathrm { b a s e } } ( F _ { \mathrm { p r i o r } } ) = 1$ If the prior family is not viable, a different family can be considered only through the rescue path. Let

$$
\mathsf { V } _ { \mathrm { r e s } } ( F ) \in \{ 0 , 1 \}\tag{44}
$$

indicate that F has the required bilateral leave-one-out support together with bilateral structural consistency. The frozen family is therefore

$$
\widehat { F } = \left\{ \begin{array} { l l } { F _ { \mathrm { p r i o r } } , } & { \mathsf { V } _ { \mathrm { b a s e } } ( F _ { \mathrm { p r i o r } } ) = 1 \ , } \\ { \mathsf { S e l e c t } _ { \mathrm { r a n k } } \left( \{ F \in \mathscr { F } _ { \mathrm { p r o p } } : \mathsf { V } _ { \mathrm { r e s } } ( F ) = 1 \} \right) , } & { \mathsf { V } _ { \mathrm { b a s e } } ( F _ { \mathrm { p r i o r } } ) = 0 \ \mathrm { a n d } \mathrm { t h e ~ s e t ~ i s ~ n o n e m p t y } \ , } \\ { \bot , } & { \mathrm { o t h e r w i s e } \ . } \end{array} \right.\tag{45}
$$

Here $\mathsf { S e l e c t } _ { \mathrm { r a n k } }$ denotes selection according to the frozen family ranking. No family is released when the required viability evidence is absent.

Assume now that $\widehat { F } \neq \bot$ . The two orientations of $\widehat F$ form the directed candidate set $\mathcal { D } ( \widehat { F } )$ . For each $e \in { \mathcal { D } } ( { \widehat { F } } )$ , let

$$
D _ { \mathrm { F E } } ( e ) \ , \quad D _ { \mathrm { F V } } ( e ) \ , \quad D _ { \mathrm { M } } ( e ) \in \{ 0 , 1 \}\tag{46}
$$

denote the FullEdge, FamilyVerify, and Mechanism direction votes. Let ${ \cal E } _ { \mathrm { M } } \in \{ 0 , 1 \}$ indicate whether the Mechanism direction evidence passes its frozen response and endpoint balance requirements.

The regular direction score is

$$
V _ { \mathrm { d i r } } ( e ) = D _ { \mathrm { F E } } ( e ) + D _ { \mathrm { F V } } ( e ) + E _ { \mathrm { M } } D _ { \mathrm { M } } ( e ) .\tag{47}
$$

A regular direction is released only when one orientation $e ^ { \star } \in { \mathcal { D } } ( { \widehat { F } } )$ is the unique orientation satisfying

$$
V _ { \mathrm { d i r } } ( e ^ { \star } ) \geq 2 .\tag{48}
$$

A separate rescue rule is available only when the Mechanism direction channel is ineligible. Let $D _ { \mathrm { { S } } } ( \bar { e } ) \in \{ 0 , 1 \}$ denote structural corroboration. Structural evidence is not a regular direction vote. If $E _ { \mathrm { M } } = 0$ , FullEdge and FamilyVerify disagree, and structural evidence corroborates the FamilyVerify orientation, the FamilyVerify orientation may be released through the rescue rule provided that the frozen family itself satisfies the rescue viability condition.

Let $e ^ { \star }$ denote the unique directed orientation accepted by either the regular rule or the rescue rule. The released transformer mapping state is

$$
( R , \widehat { P } ) = \left\{ \begin{array} { l l } { ( 1 , e ^ { \star } ) , } & { \widehat { F } \neq \bot \mathrm { ~ a n d ~ a ~ u n i q u e ~ d i r e c t i o n ~ i s ~ a c c e p t e d ~ } , } \\ { ( 0 , \bot ) , } & { \mathrm { ~ o t h e r w i s e ~ } . } \end{array} \right.\tag{49}
$$

Proposition 8 (Transformer fail-closed mapping contract). The transformer builder releases a directed pair only if a viable family is frozen and one of its two orientations satisfies the regular direction rule or the rescue rule above. If family viability is insufficient, if direction evidence is insufficient, or if the accepted orientation is not unique, then $R = 0$ and $\widehat { P } = \bot$ . Once the direction firewall has rejected the frozen family, no later family fallback can create an active mapping.

Proof. Equation 45 returns either a viable frozen family or ⊥. Conditional on a viable family, Equation 49 sets $R = 1$ only when exactly one orientation satisfies an authorized direction rule. Every other case returns $( R , { \widehat { P } } ) = ( 0 , \bot )$ . Structural evidence can corroborate the rescue rule but does not contribute to the regular direction vote in Equation 47. The absence of a later family fallback completes the fail-closed alternative. 口

This proposition is a deterministic release contract for the observable builder state. It does not claim that the released pair equals the unknown true attacked pair. Ground truth, attack provenance, and the oracle attacked pair are not inputs to the release rule.

## A.1.7 REPAIR-STATE SAFETY

Let $\mathcal { C } _ { 0 }$ be the provisional reference population after architecture-specific cross-fitting and let $V _ { i } \in$ {0, 1} denote the victim-safety veto. Define

$$
\mathcal { C } _ { \mathrm { s a f e } } = \{ i \in \mathcal { C } _ { 0 } : V _ { i } = 0 \} \ .\tag{50}
$$

Proposition 9 (One-way safety). Every later capacity-controlled repair bank satisfies

$$
{ \mathcal C } _ { \mathrm { b a n k } } \subseteq { \mathcal C } _ { \mathrm { s a f e } } \subseteq { \mathcal C } _ { 0 } .\tag{51}
$$

Consequently, a vetoed reference cannot be reintroduced after the safety decision.

Proof. The first inclusion is the construction rule for all later banks and fixed points. The second follows directly from the definition of $\mathcal { C } _ { \mathrm { s a f e } }$ □

This guarantees monotone selection, not true clean provenance. If required reference count or semantic coverage cannot be maintained, the method abstains.

## A.1.8 ORTHOGONAL SEMANTIC MATCHING

Let $u ( x )$ be the architecture-specific semantic coordinate used by the probability reference. Assume distinct victim and target centers $\mu _ { v } ^ { u } \neq \mu _ { t } ^ { u }$ and define

$$
a _ { v t } = \frac { \mu _ { t } ^ { u } - \mu _ { v } ^ { u } } { \Vert \mu _ { t } ^ { u } - \mu _ { v } ^ { u } \Vert _ { 2 } } , \Pi _ { v t } ^ { \perp } = I - a _ { v t } a _ { v t } ^ { \top } .\tag{52}
$$

Lemma 10 (Orthogonal matching ignores pair-axis displacement). For semantic coordinates $z , r$ and scalar $c ,$

$$
\Pi _ { v t } ^ { \perp } [ ( z + c a _ { v t } ) - r ] = \Pi _ { v t } ^ { \perp } ( z - r ) .\tag{53}
$$

Proof. Since $\begin{array} { r } { \| a _ { v t } \| _ { 2 } = 1 , \Pi _ { v t } ^ { \bot } a _ { v t } = ( I - a _ { v t } a _ { v t } ^ { \top } ) a _ { v t } = 0 . } \end{array}$

Thus orthogonal retrieval is exactly invariant to displacement along the chosen victim–target axis. It does not assume that a real attack lies entirely on that axis.

## A.1.9 MINIMAL VICTIM-TARGET LOGIT CORRECTION.

Consider a residual pair-logit substep applied after any feature transport. For pre-correction logits $\boldsymbol { \ell } \in \mathbb { R } ^ { K }$ , let $q = \operatorname { s o f t m a x } ( \bar { \ell } )$ and define

$$
m _ { \ell } = \ell _ { v } - \ell _ { t } = \log \frac { q _ { v } } { q _ { t } } .\tag{54}
$$

Let $m _ { v t } ^ { \mathrm { r e p } } > 0$ be the frozen required victim-over-target margin, calibrated from safety-approved victim reference pixels.

Proposition 11 (Minimum- ${ \boldsymbol { \cdot } } { \boldsymbol { \ell } } _ { 2 }$ full pair correction). Among corrections $\Delta \in \mathbb { R } ^ { K }$ satisfying

$$
\Delta _ { k } = 0 \ , \quad ( k \notin \{ v , t \} ) \ ,\tag{55}
$$

$$
\Delta _ { v } + \Delta _ { t } = 0 ,\tag{56}
$$

$$
( \ell _ { v } + \Delta _ { v } ) - ( \ell _ { t } + \Delta _ { t } ) \geq m _ { v t } ^ { \mathrm { r e p } } ,\tag{57}
$$

the unique minimum- ${ \boldsymbol { \cdot } } { \boldsymbol { \ell } } _ { 2 }$ correction is

$$
\Delta _ { v } = \frac { d } { 2 } , \qquad \Delta _ { t } = - \frac { d } { 2 } , \qquad d = [ m _ { v t } ^ { \mathrm { r e p } } - ( \ell _ { v } - \ell _ { t } ) ] _ { + } .\tag{58}
$$

Proof. Write $r = m _ { v t } ^ { \mathrm { r e p } } - ( \ell _ { v } - \ell _ { t } )$ . The zero-sum constraint gives $\Delta _ { t } = - \Delta _ { v } .$ , so feasibility is $2 \Delta _ { v } \geq r$ and $\lVert \Delta \rVert _ { 2 } ^ { 2 } \stackrel { \sim } { = } 2 \Delta _ { v } ^ { 2 }$ . If $r \leq 0 , \Delta _ { v } = 0$ is feasible and uniquely minimizes the norm. If $r > 0 .$ , the smallest feasible value is $\Delta _ { v } = r / 2 .$ . Combining both cases yields Equation 58. □

Corollary 12 (Effect of the full pair correction on odds). Let q and $q ^ { \prime }$ be the softmax distributions immediately before and after thefull correction in Proposition 11. Then

$$
\frac { q _ { v } ^ { \prime } / q _ { t } ^ { \prime } } { q _ { v } / q _ { t } } = e ^ { d } ~ .\tag{59}
$$

The actual runtime may use a branch-dependent fraction of this residual correction and may subsequently apply a fail-closed projection. Feature transport can also change logits of classes outside $\{ v , t \}$ indirectly, i.e., the “pair-only” statement applies only to this explicit residual-logit substep.

## A.1.10 RESIDUAL ATTACK SUCCESS

Proposition 13 (Residual-ASR bound). Under the definitions in Section A.1.1,

$$
\mathrm { A S R } _ { \mathrm { d e f } } \leq \mathrm { A S R } _ { \mathrm { b a s e } } \left[ \eta _ { \mathrm { l o c } } + ( 1 - \eta _ { \mathrm { l o c } } ) \varepsilon _ { \mathrm { r e p } } \right] + \delta _ { \mathrm { n e w } } \ .\tag{60}
$$

Proof. Partition $\{ { \widehat { Y } } ^ { \mathrm { d e f } } = t \}$ into A and $A ^ { c }$ . The $A ^ { c }$ contribution is exactly $\delta _ { \mathrm { n e w } } . \mathrm { I f } \mathbb { P } _ { v } ( A ) = 0$ , the A contribution is zero and the claim follows. Otherwise, partition A by M. On $A \cap \{ M = 0 \}$ , the defended-target event has probability at most $\mathrm { A S R } _ { \mathrm { b a s e } } \eta _ { \mathrm { l o c } }$ . On $A \cap \{ M = 1 \}$ , the exact contribution is $\mathrm { A S R } _ { \mathrm { b a s e } } ( \bar { 1 } - \eta _ { \mathrm { l o c } } ) \varepsilon _ { \mathrm { r e p } } .$ , including the zero-mass convention of Equation 20. Adding the disjoint contributions proves the bound. □

Corollary 14 (Reference-aware residual-ASR bound). Assume $C _ { \mathrm { p a i r } }$ and Equation 17. Then

$$
\mathrm { A S R } _ { \mathrm { d e f } } \leq \mathrm { A S R } _ { \mathrm { b a s e } } \left[ \eta _ { \mathrm { r e f } } + ( 1 - \eta _ { \mathrm { r e f } } ) \varepsilon _ { \mathrm { r e p } } \right] + \delta _ { \mathrm { n e w } } \ .\tag{61}
$$

Proof. Proposition 6 gives $\eta _ { \mathrm { l o c } } \leq \eta _ { \mathrm { r e f } }$ . For fixed $\varepsilon _ { \mathrm { r e p } } \in [ 0 , 1 ]$ , the function $f ( \eta ) = \eta + ( 1 - \eta ) \varepsilon _ { \mathrm { r e p } }$ is nondecreasing because $f ^ { \prime } ( \eta ) = 1 - \varepsilon _ { \mathrm { r e p } } \geq 0$ . Substitute this inequality into Proposition 13.

## A.1.11 EFFECT OF THE NO NEW TARGET VETO

Let $\begin{array} { r } { S = \{ x : M ( x ) = 1 \} } \end{array}$ denote the supported repair region. The No New Target Veto is applied to the discovered target t<sup>ˆ</sup>and enforces

$$
x \not \in { \mathcal { S } } \ , \quad \widehat { Y } ^ { \mathrm { h e a d } } ( x ) \not = \widehat { t } \quad \Longrightarrow \quad \widehat { Y } ^ { \mathrm { d e f } } ( x ) \not = \widehat { t } \ .\tag{62}
$$

Proposition 15 (New-target support restriction). Assume $C _ { \mathrm { t g t } } , i . e .$ , an active state is released and the discovered target satisfies $\hat { t } = t$ . Within $\mathbb { P } _ { v }$

$$
\begin{array} { r } { \delta _ { \mathrm { n e w } } \leq \mathbb { P } _ { v } ( x \in { \mathcal S } , A ^ { c } ) \ . } \end{array}\tag{63}
$$

$$
I f S \subseteq A , t h e n \ \delta _ { \mathrm { n e w } } = 0 .
$$

Proof. Under $C _ { \mathrm { t g t } } , \hat { t } = t ,$ , so Equation 62 excludes $\{ \widehat { Y } ^ { \mathrm { d e f } } = t , A ^ { c } \}$ outside ${ \mathcal { S } } .$ Hence any event counted by $\delta _ { \mathrm { n e w } }$ must lie in $\mathcal { S } \cap \bar { \boldsymbol { A } } ^ { c }$ . If ${ \mathcal { S } } \subseteq A$ , this set is empty. □

Corollary 16 (No new true-target attacks under target-correct transformer localization). Assume $C _ { \mathrm { t g t } }$ . Then the transformer runtime satisfies

$$
M _ { \mathrm { T F } } \subseteq \{ \widehat { Y } ^ { \mathrm { h e a d } } = \widehat { t } \} = \{ \widehat { Y } ^ { \mathrm { h e a d } } = t \} = A \ .
$$

Consequently,

$$
\delta _ { \mathrm { n e w } } = 0 .\tag{64}
$$

Proof. Every transformer localization branch is restricted to baseline pixels predicted as the discovered target t<sup>ˆ</sup>, and the final localization mask is their union. Hence $\bar { M } _ { \mathrm { T F } } \subseteq \{ \widehat { Y } ^ { \mathrm { h e a d } } = \widehat { t } \}$ . Under $C _ { \mathrm { t g t } }$ we have $\hat { t } = t , \mathrm { s o } M _ { \mathrm { T F } } \subseteq A$ . Proposition 15 therefore gives $\delta _ { \mathrm { n e w } } = 0$ □

## A.1.12 CLEAN-SIDE INTERVENTION

Define

$$
\kappa _ { \mathrm { c l } } = \mathbb { P } _ { \mathrm { c l } } ( M = 1 )\tag{65}
$$

and

$$
\delta _ { \mathrm { s p i l l } } ^ { \mathrm { c l } } = \mathbb { P } _ { \mathrm { c l } } ( \widehat { Y } ^ { \mathrm { d e f } } \neq \widehat { Y } ^ { \mathrm { h e a d } } , M = 0 ) \ .\tag{66}
$$

Proposition 17 (Clean prediction-change bound).

$$
\mathbb { P } _ { \mathrm { c l } } ( \widehat { Y } ^ { \mathrm { d e f } } \neq \widehat { Y } ^ { \mathrm { h e a d } } ) \le \kappa _ { \mathrm { c l } } + \delta _ { \mathrm { s p i l l } } ^ { \mathrm { c l } } .\tag{67}
$$

Proof. Partition the prediction-change event by M. The localized contribution is at most $\mathbb { P } _ { \mathrm { c l } } ( M =$ $1 ) = \kappa _ { \mathrm { c l } }$ and the non-localized contribution is exactly $ { \delta _ { \mathrm { s p i l l } } } ^ { \mathrm { c l } }$ □

Exact preservation follows when both terms are zero, but this is not assumed. The probabilityresidual branch is calibrated under a finite empirical activation budget on the safety-approved reference population. This budget does not constitute a population-level guarantee on future clean data.

## A.1.13 DEPENDENCY SUMMARY

Table 2: Role and scope of the theoretical results.
<table><tr><td>Result</td><td>Main condition</td><td>Consequence</td></tr><tr><td>Lemma 1</td><td>local directed transport</td><td>pair margin decreases</td></tr><tr><td>Cor. 2</td><td>positive initial pair margin</td><td>pairwise preference flips</td></tr><tr><td>Lemma 3</td><td>fixed feasible transport amount</td><td>local inverse exists</td></tr><tr><td>Lemma 4</td><td>feasible one-edge shift</td><td>coupling representation</td></tr><tr><td>Lemma 5</td><td>evidence definition</td><td>surviving victim evidence</td></tr><tr><td>Prop. 6</td><td>pair-correct state and additive semantic arm</td><td> $\eta _ { \mathrm { l o c } } \leq \eta _ { \mathrm { r e f } }$ </td></tr><tr><td>Prop. 7</td><td>scalar evidence gap</td><td>sufficient scalar separation</td></tr><tr><td>Prop. 8</td><td>family viability and direction agreement</td><td>fail-closed mapping release</td></tr><tr><td>Prop. 9</td><td>monotone filtering</td><td>vetoed references cannot return</td></tr><tr><td>Lemma 10</td><td>nondegenerate pair axis</td><td>axis-invariant matching</td></tr><tr><td>Prop. 11</td><td>full pair-only correction</td><td>minimum  $\boldsymbol { \cdot } \boldsymbol { \ell } _ { 2 }$  substep</td></tr><tr><td>Prop. 13</td><td>localization/repair terms</td><td>residual-ASR bound</td></tr><tr><td>Cor. 16</td><td>target-correct state and transformer support  $\subseteq A$ </td><td> $\delta _ { \mathrm { n e w } } = 0$ </td></tr><tr><td>Prop. 17</td><td>selective intervention</td><td>clean-change bound</td></tr></table>

Remark 18 (Scope of the theory). The analysis does not claim that a real backdoor changes only two class probabilities, that its internal effect is one-dimensional, or that the unknown attack operator is observed exactly. The inverse operator and the transposed coupling are distinct mathematical objects, as stated in the main text and formalized above. It also does not claim that the safety-approved reference population is genuinely clean, or that a scalar population pair score reproduces the complete transformers mapping selector. The transport model is a local abstraction and the coupling is feasible rather than optimal. Pair-separation results are sufficient, not necessary. Proposition 8 states a deterministic release contract conditional on observable builder evidence and is not a distributionfree theorem that the released pair equals the true attacked pair. Calibration budgets are empirical reference-population constraints. Clean preservation depends on localization and spillover rather than on an oracle clean-input gate.

## A.2 ADDITIONAL METHOD DETAILS

This subsection specifies the implementation details required to reproduce the operational pipeline in Section 4. Throughout this section, (v, t) denotes the operational victim-target pair released by the attack-unaware builder. In the preceding theoretical analysis, (v, t) instead denotes the true attacked pair, while (ˆv, t<sup>ˆ</sup>) denotes the released pair.

We use safety-approved or clean-like reference rather than clean reference, since the defense has no access to true clean/poison provenance.

## A.2.1 INFORMATION BOUNDARY AND STATE CONSTRUCTION

RePAIR receives a frozen segmentation model and an unlabeled image pool that may contain triggered samples. Semantic ground truth, clean/poison provenance, attack masks, clean twins, attack identity, poisoning rate, and the true victim–target pair are excluded from all defense decisions. Annotation fields are removed before every forward pass contributing to pair discovery, reference construction, calibration, localization, or repair. Oracle information is accessed only after the official state has been frozen and serialized and is used exclusively for evaluation and diagnostics.

State construction proceeds through three fail-closed stages: attack-unaware mapping inference, construction of an image-disjoint safety-approved reference population, and reconstruction of the frozen runtime state from that population. Failure to establish sufficient mapping confidence, reference capacity, or semantic coverage leaves the defense inactive.

## A.2.2 ARCHITECTURE-SPECIFIC SEMANTIC EVIDENCE

For CNNs, normalized features from the final backbone representation define the CNN semantic reference. For head-pseudo-labeled reference pixels $\mathcal { P } _ { k } .$ , class centroids are

$$
\mu _ { k } = \frac { 1 } { | { \mathcal P } _ { k } | } \sum _ { x \in { \mathcal P } _ { k } } z ( x ) \mathrm { ~ , ~ } \qquad \widehat { Y } ^ { \mathrm { \scriptsize { p r o t } } } ( x ) = \arg \operatorname* { m a x } _ { k } \sin ( z ( x ) , \mu _ { k } ) .\tag{68}
$$

Directed internal-to-head disagreement for candidate pair $( v , t )$ is measured by

$$
G _ { i } ^ { \mathrm { C N N } } ( v , t ) = \frac { 1 } { | \Omega _ { i } | } \sum _ { x \in \Omega _ { i } } \mathbf { 1 } \widehat { Y } ^ { \mathrm { p r o t } } ( x ) = v \wedge \widehat { Y } ^ { \mathrm { h e a d } } ( x ) = t \ .\tag{69}
$$

The final builder resolves the target before selecting the victim. Once the target is confidenceresolved, victim proposals are restricted to that target and combine the established pair-evidence score with complementary victim-retention and victim-interior cues. A proposal is accepted only when it is sufficiently separated from competing candidates, otherwise mapping fails closed. No class identifier or attack-specific pair is hard-coded. Every selected pair must subsequently pass the unchanged reference-safety pipeline.

Transformers combine semantic evidence from L selected internal representations. For representation ℓ, victim and target centroid similarities are

$$
c _ { \ell , v } ( x ) = \sin ( z _ { \ell } ( x ) , \mu _ { \ell , v } ) , c _ { \ell , t } ( x ) = \sin ( z _ { \ell } ( x ) , \mu _ { \ell , t } ) .\tag{70}
$$

The selected layers are concatenated into

$$
\phi _ { v , t } ( \boldsymbol { x } ) = \left[ c _ { 1 , v } ( \boldsymbol { x } ) , \ldots , c _ { L , v } ( \boldsymbol { x } ) , c _ { 1 , t } ( \boldsymbol { x } ) , \ldots , c _ { L , t } ( \boldsymbol { x } ) \right] ^ { \top } ,\tag{71}
$$

and a regularized logistic readout

$$
r _ { v , t } ( \boldsymbol { x } ) = \sigma \left( \boldsymbol { w } ^ { \top } \widetilde { \phi } _ { v , t } ( \boldsymbol { x } ) + b \right)\tag{72}
$$

provides the internal target compatibility. where w and b denote the learned weight vector and bias of the regularized logistic readout, respectively, and $\widetilde { \phi } _ { v , t } ( \boldsymbol { x } )$ is the standardized joint multi-layer feature representation for pair (v, t). The corresponding victim support is

$$
p _ { v } ^ { \mathrm { i n t } } ( x ) = 1 - r _ { v , t } ( x ) \ .\tag{73}
$$

Training labels for this readout are generated exclusively from frozen head pseudo-labels.

Transformers mapping first forms stable undirected family proposals from structural, mechanism, and family-verification evidence. A frozen rank prior is retained whenever its candidate passes the base viability test. If it is not viable, replacement is permitted only through the rescue path, which requires bilateral leave-one-out support together with bilateral structural consistency. Direction is then authorized within the frozen family using FullEdge, FamilyVerify, and an eligible Mechanism vote. Structural evidence does not act as a regular direction vote and is used only to corroborate the rescue rule. Insufficient family viability or direction evidence leaves the mapping inactive. The formal release contract is given in Proposition 8.

## A.2.3 REFERENCE-STATE SAFETY

Let $\mathcal { C } _ { 0 }$ denote the provisional reference population and $V _ { i } \in \{ 0 , 1 \}$ the victim-safety veto. The safety-approved population is

$$
\mathcal { C } _ { \mathrm { s a f e } } = \{ i \in \mathcal { C } _ { 0 } : V _ { i } = 0 \} \ .\tag{74}
$$

All later repair banks satisfy

$$
{ \mathcal C } _ { \mathrm { b a n k } } \subseteq { \mathcal C } _ { \mathrm { s a f e } } \subseteq { \mathcal C } _ { 0 } .\tag{75}
$$

Thus, a vetoed reference cannot be reintroduced by refinement, capacity restriction, fallback, or backfilling.

Safety statistics are estimated image-disjointly. References with severe victim-evidence collapse, or moderate collapse accompanied by unusually strong victim-to-target rupture, are rejected. State construction requires at least 24 safety-approved images and at least 75% semantic-class coverage.

Capacity restriction is also fail-closed. If a compressed reference core loses sufficient victim or target prototype support, the implementation may retry the same capacity check on the larger already safety-approved reservoir. This fallback does not relax thresholds, re-admit vetoed samples, or introduce backfill.

## A.2.4 LOCALIZATION AND REPAIR

For CNNs, runtime localization combines direct semantic contradiction, reference-calibrated geometric residuals, and a semantic probability residual:

$$
M _ { \mathrm { C N N } } = M _ { \mathrm { s e m a n t i c } } \vee M _ { \mathrm { g e o m } } \vee M _ { \mathrm { p r o b } } .\tag{76}
$$

For transformers, the final support is the sequential union

$$
M _ { \mathrm { T F } } = M _ { \mathrm { s e m a n t i c } } \lor M _ { \mathrm { s r c } } \lor M _ { \mathrm { t g t } } \lor M _ { \mathrm { t r a j } } \lor M _ { \mathrm { p r o b } } ~ ,\tag{77}
$$

with

$$
M _ { \mathrm { T F } } \subseteq \{ \widehat { Y } ^ { \mathrm { h e a d } } = t \} .\tag{78}
$$

Both architectures use the same final probability-residual principle. For discovered pair $( v , t )$ , define the target-over-victim probability excess

$$
d _ { p } ( x ) = q _ { t } ( x ) - q _ { v } ( x ) .\tag{79}
$$

Relative to the expected excess $\widehat { d } _ { \mathrm { r e f } } ( x )$ obtained from the safety-approved semantic reference,

$$
D _ { \mathrm { p r o b } } ( x ) = \Big [ d _ { p } ( x ) - \widehat { d } _ { \mathrm { r e f } } ( x ) \Big ] _ { + } \enspace .\tag{80}
$$

The residual, $\delta _ { \mathrm { p r o b } } .$ , is calibrated in four image-disjoint folds under the fixed empirical referencepixel budget.

CNNs repair localized features from the final backbone layer toward matched safety-approved victim references and rerun the unchanged remainder of the network. Transformers transports localized pixels toward its frozen multi-layer victim manifold. Both architectures may subsequently apply a conservative pair-specific output correction. No network parameter is retrained or permanently modified.

Finally, feature repair may influence predictions outside directly localized cells. Let $\widehat { Y } ^ { \mathrm { r e p } }$ denote the prediction immediately before the output veto and $S = \{ x : \overset { \cdot } { M } ( x ) = 1 \}$ . RePAIR reverts to the undefended prediction whenever

$$
x \not \in { \mathcal { S } } \ , \qquad { \widehat { Y } } ^ { \mathrm { h e a d } } ( x ) \not = t \ , \qquad { \widehat { Y } } ^ { \mathrm { r e p } } ( x ) = t \ .\tag{81}
$$

Thus, the repair cannot introduce a new unsupported target prediction outside the localized support.

## A.3 DETAILS ON EXPERIMENTAL SETTING

Datasets. We consider three datasets for evaluation, namely Cityscapes Cordts et al. (2016), ADE20K (et al., 2017), and PASCAL VOC 2012 (Everingham et al., 2015). Cityscapes consists of 2,975 training and 500 validation images with a fixed resolution of 2048 × 1024 pixels and defines 19 semantic classes for evaluation. The dataset focuses on densely annotated urban street scenes and contains a diverse set of simultaneously occurring object and stuff classes, such as roads, buildings, vegetation, pedestrians, and vehicles.

ADE20K is a substantially larger and more scene-centric benchmark, comprising 20,210 training and 2,000 validation images with variable native resolutions and dense pixel-level annotations across 150 semantic classes, including both object and stuff categories.

We further evaluate RePAIR on PASCAL VOC 2012, whose semantic segmentation benchmark comprises 1,464 training and 1,449 validation images with variable image resolutions, typically around 500 × 375 pixels, and defines 20 foreground classes plus a background class. In contrast to the more scene-centric Cityscapes and ADE20K datasets, PASCAL VOC is predominantly objectcentric: images generally focus on one or more salient foreground objects belonging to the predefined object classes, while pixels not assigned to these classes are represented by a single background label.

Since both, PASCAL VOC and ADE20K, contain images with varying native resolutions, we resize all images to 512 × 512 pixels. We use ADE20K as an attack-effectiveness negative control. since the evaluated attacks do not produce a sufficiently strong backdoor effect on this dataset, RePAIR is not applied.

Segmentation models. We evaluate two representative segmentation architectures: DeepLabV3+ with a ResNet-50 backbone (Chen et al., 2018) and SegFormer with a MiT-B5 backbone (Xie et al., 2021). DeepLabV3+ is trained for 80,000 iterations on Cityscapes and ADE20K and for 40,000 iterations on PASCAL VOC 2012, while SegFormer is trained for 160,000 iterations on all datasets. The other training parameters are as specified in the repository<sup>1</sup>.

Backdoor attacks. We consider IBA with NNI and PRL (Lan et al., 2024), ConSeg (Abbasi et al., 2025), OFBA (Mao et al., 2023), and BadSeg (Zhang et al., 2026). All attacks are trained with a 10% poisoning rate, and trigger configurations follow the corresponding attack papers.

For IBA, we use the Hello Kitty trigger and the victim-to-target relabeling described in the original method. IBA-NNI places the trigger on non-victim pixels close to the victim region, using a 15 × 15 trigger on PASCAL VOC and a 55 × 55 trigger on Cityscapes. IBA-PRL additionally applies Pixel Random Labeling (PRL), where 50,000 additional pixels are assigned randomly sampled labels from the same segmentation mask.

OFBA uses a 29 × 29 black-and-white checkerboard trigger with visibility coefficient λ = 0.4, placed entirely within an object region. The corresponding object is relabeled to the target class, while the attacked object and trigger position are varied during training.

ConSeg uses a naturally occurring object as a semantic trigger and relabels the victim class as the target class. It additionally modifies victim-region labels using target-related co-occurring classes, with five such classes and four replaced pixels per class in the default configuration.

BadSeg jointly optimizes the trigger and victim-target pair using a surrogate segmentation model. We evaluate its object-to-object (O20), object-to-background (O2B), background-to-object (B2O), and background-to-background (B2B) configurations.

Baseline defenses. For the defense benchmarks, we use the method-specific parameters listed in Table 3, following their original configurations.

Evaluation metrics. We evaluate attack effectiveness and segmentation utility using Attack Success Rate (ASR), benign mean intersection-over-union (mIoU-B), and attacked mIoU (mIoU-A). ASR measures the fraction of ground-truth victim-class pixels on triggered inputs that are predicted as the attacker-selected target class. For BadSeg-B2O, where the attack is restricted to a designated spatial region, ASR is computed only within the corresponding evaluation region.

Table 3: Hyperparameters of the benchmark defenses.
<table><tr><td>Defense</td><td>Parameter</td><td>Value</td></tr><tr><td rowspan="2">Fine-tuning</td><td>Clean data fraction</td><td>10%</td></tr><tr><td>Epochs</td><td>10</td></tr><tr><td rowspan="2">Pruning</td><td>Pruning ratio</td><td>10%</td></tr><tr><td>Criterion</td><td>Least-active final-layer channels</td></tr><tr><td rowspan="9">ABL</td><td>Isolation epochs</td><td>20</td></tr><tr><td>Isolation ratio p</td><td>0.01</td></tr><tr><td>γ</td><td>0.5</td></tr><tr><td>Batch / scoring batch</td><td>8/8</td></tr><tr><td>Unlearning epochs</td><td>5</td></tr><tr><td>Unlearning optimizer</td><td>SGD</td></tr><tr><td>Unlearning learning rate</td><td> $5 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Recovery epochs</td><td>0</td></tr><tr><td>Clean-loss guard</td><td>32 samples, threshold 3×</td></tr><tr><td rowspan="9">Segment Cleanse</td><td>Clean images</td><td>1000</td></tr><tr><td>Working resolution</td><td> $5 1 2 \times 1 0 2 4$ </td></tr><tr><td>Target knowledge</td><td>Unknown; all alternatives scanned</td></tr><tr><td>Reverse optimizer</td><td>Adam</td></tr><tr><td>Reverse learning rate</td><td> $5 \times 1 0 ^ { - 3 }$ </td></tr><tr><td>Regularization α</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td>Maximum reverse epochs</td><td>100</td></tr><tr><td>Early-stop warm-up / patience</td><td> $1 0 / 5$ </td></tr><tr><td>Relative threshold</td><td> $1 0 ^ { \dot { - } 4 }$ </td></tr><tr><td></td><td>Parallel target candidates</td><td>2</td></tr><tr><td></td><td>Mitigation</td><td>Decoder-only</td></tr><tr><td></td><td>Mitigation optimizer</td><td>SGD</td></tr><tr><td></td><td>Mitigation learning rate</td><td> $1 0 ^ { - 4 }$ </td></tr><tr><td></td><td>Mitigation epochs</td><td>20</td></tr></table>

mIoU-A measures the mean IoU on triggered inputs after excluding the pixels used to evaluate the attack. For class-wide attacks, all ground-truth victim-class pixels are excluded, whereas for BadSeg-B2O only the designated attack region is excluded. mIoU-B measures the mean IoU on clean inputs. In the literature, mIoU-A and mIoU-B are also referred to as Poisoned Benign Accuracy (PBA) and Clean Benign Accuracy (CBA), respectively (Lan et al., 2024; Abbasi et al., 2025).

Lower ASR indicates stronger attack mitigation, while higher mIoU-A and mIoU-B indicate better segmentation utility.

Hyperparameter selection. All RePAIR hyperparameters are treated as global method settings and fixed independently of the evaluated attack instance. We do not select parameters based on attack identity, the oracle victim-target pair, clean/poison provenance, attack masks, ground-truth segmentation labels, or evaluation ASR. Structural parameters, such as minimum reference capacity, semantic coverage requirements, retrieval sizes, and calibration budgets, are fixed across attacks. Data-dependent thresholds are reconstructed only from the frozen safety-approved reference population using image-disjoint calibration and remain fixed during evaluation. Thus, attack-specific test performance is never used to tune the repair state.

Run-specific preprocessing, checkpoints, random seeds, and execution parameters are provided in the experiment configurations summarized in Table 4.

## A.4 ADDITIONAL EXPERIMENTAL RESULTS

ADE20K. ADE20K contains 150 semantic classes and constitutes a substantially different semantic setting from Cityscapes and PASCAL VOC. In our experiments, the evaluated attack configurations do not establish a meaningful backdoor effect in the undefended ADE20K models. The victim-target pair bein used on ADE20K is car-road.

Table 4: Principal global state-construction and inference parameters for Cityscapes.
<table><tr><td>Parameter</td><td>Variable</td><td>Value</td></tr><tr><td>Shared</td><td></td><td></td></tr><tr><td>Image-disjoint safety/calibration folds</td><td> $K _ { \mathrm { f o l d } }$ </td><td>4</td></tr><tr><td>Minimum safety-approved images</td><td> $N _ { \mathrm { s a f e } } ^ { \mathrm { m i n } }$ </td><td>24</td></tr><tr><td>Minimum semantic class coverage</td><td> $\gamma _ { \mathrm { c o v } }$ </td><td>75%</td></tr><tr><td>Semantic probability budget</td><td> $\delta _ { \mathrm { p r o b } }$ </td><td>0.05</td></tr><tr><td>DeeplabV3+</td><td></td><td></td></tr><tr><td>Target-authority winner ratio</td><td> $\tau _ { \mathrm { t g t } }$ </td><td>4.00</td></tr><tr><td>Legacy victim winner ratio</td><td> $\tau _ { \mathrm { v i c } }$   $\mathord { \sim } ^ { \mathrm { m i n } }$ </td><td>1.20</td></tr><tr><td>Legacy victim robust MAD-z</td><td> $z _ { \mathrm { v i c } } ^ { \mathrm { - } }$   $\gamma ^ { \mathrm { { \dot { m i n } } } }$ </td><td>1.00</td></tr><tr><td>Geometry-fallback robust MAD-z Reference-bank cap (per relevant class)</td><td> $z _ { \mathrm { g e o } }$   $N ^ { \mathrm { { m a x } } }$ </td><td>2.00</td></tr><tr><td>Self-counterfactual strength</td><td> $N _ { \mathrm { r e f } } ^ { \mathrm { n u c . } }$ </td><td>4096</td></tr><tr><td></td><td> $\lambda _ { \mathrm { c f } }$ </td><td>0.25</td></tr><tr><td>Base residual allowance</td><td> $\delta _ { \mathrm { b a s e } }$ </td><td>0.02</td></tr><tr><td>Auxiliary residual allowance</td><td> $\delta _ { \mathrm { a u x } }$ </td><td>0.005</td></tr><tr><td>SegFormer</td><td></td><td></td></tr><tr><td>Semantic victim-support quantile</td><td> $q _ { \mathrm { s e m } }$ </td><td>0.995</td></tr><tr><td>Repair-margin quantile</td><td> $q _ { \mathrm { r e p } }$ </td><td>0.10</td></tr><tr><td>Repair-margin quantile</td><td> $q _ { \mathrm { r e p } }$ </td><td>0.10</td></tr><tr><td>Minimum suspicious OOF group (images)</td><td> $N _ { \mathrm { O O F } } ^ { \mathrm { m i n } }$ </td><td>12</td></tr><tr><td>Manifold retrieval neighbors</td><td> $k _ { \mathrm { m a n } }$ </td><td>8</td></tr><tr><td>Trajectory residual budget</td><td> $\delta _ { \mathrm { t r a j } }$ </td><td>0.005</td></tr></table>

Table 5: ADE20K Attack Results Undefended attack success rate (ASR, %). RePAIR is not applied because the evaluated attacks do not establish a meaningful backdoor effect.
<table><tr><td>Model</td><td>Method</td><td>IBA (NNI)</td><td>IBA (PRL)</td><td>ConSeg</td><td>OFBA</td><td>BadSeg-O2B</td></tr><tr><td>DeepLabV3+</td><td>no defense</td><td>3.02</td><td>3.22</td><td>2.88</td><td>2.40</td><td>2.58</td></tr><tr><td>SegFormer</td><td>no defense</td><td>2.51</td><td>2.38</td><td>4.93</td><td>2.48</td><td>2.43</td></tr></table>

We therefore do not treat ADE20K as an additional defense benchmark. A defense result is only informative when the underlying attack first achieves a substantive victim-to-target effect, reducing an already negligible ASR would not demonstrate successful backdoor removal.

Accordingly, the official ADE20K evaluation is baseline-only. RePAIR state construction and repair are not applied. We report the undefended ASR to document the attack effectiveness of the evaluated configurations.

The ADE20K results should therefore be interpreted as an attack-effectiveness control rather than as evidence of defense performance. They show that, under the evaluated configurations, the prerequisite backdoor behavior does not emerge strongly enough to support a meaningful mitigation experiment.

PASCAL VOC 2012. PASCAL VOC provides a challenging transfer setting because the background class occupies a large fraction of many images and represents a heterogeneous collection of visual content rather than a compact semantic object class. Consequently, natural victim-tobackground relations can compete with the directed evidence produced by an O2B backdoor.

Unlike the Cityscapes experiments, all attacks considered on VOC use the same O2B-style semantic relation. The victim–target pair is used only to compute the reported ASR and is never supplied to RePAIR. The chosen victim-target pair for VOC is person-background.

Table 6 evaluates whether the attack-unaware state builder can recover a usable repair state despite the strongly background-dominated semantic distribution of VOC. The reported defended values correspond only to states released by the fail-closed construction procedure. No oracle victim or target information is used during state construction.

Table 6: PASCAL VOC 2012. Undefended and defended attack success rate (ASR, %).
<table><tr><td>Model</td><td>Attack</td><td>No defense</td><td>RePAIR</td></tr><tr><td rowspan="5">DeeplabV3+</td><td>IBA (NNI)</td><td>12.21</td><td>5.41</td></tr><tr><td>IBA (PRL)</td><td>15.03</td><td>5.45</td></tr><tr><td>ConSeg</td><td>16.12</td><td>7.05</td></tr><tr><td>OFBA</td><td>99.71</td><td>8.51</td></tr><tr><td>BADSEG-O2B</td><td>92.14</td><td>19.67</td></tr><tr><td rowspan="5">SegFormer</td><td>IBA (NNI)</td><td>4.83</td><td>0.12</td></tr><tr><td>IBA (PRL)</td><td>7.26</td><td>0.19</td></tr><tr><td>ConSeg</td><td>6.67</td><td>0.15</td></tr><tr><td>OFBA</td><td>98.42</td><td>0.50</td></tr><tr><td>BADSEG-O2B</td><td>95.78</td><td>0.34</td></tr></table>

Table 7: PASCAL VOC 2012 segmentation utility. Mean IoU on clean inputs (mIoU-B) and triggered inputs (mIoU-A).
<table><tr><td>Model</td><td>mIoU-A</td><td>mIoU-B</td></tr><tr><td>DeeplabV3+ / ResNet-50</td><td> $6 6 . 2 4 _ { - 4 . 9 9 } ^ { + 2 . 7 1 }$ </td><td> $5 8 . 2 1 _ { - 2 . 0 5 } ^ { + 1 . 4 8 }$ </td></tr><tr><td>SegFormer / MiT-B5</td><td> $8 4 . 4 5 _ { - 3 . 3 4 } ^ { + 1 . 2 8 }$ </td><td> $7 6 . 1 7 _ { - 0 . 6 2 } ^ { + 0 . 2 2 }$ </td></tr></table>

To quantify the utility retained after repair, we additionally report clean and attacked segmentation performance in Table 7.