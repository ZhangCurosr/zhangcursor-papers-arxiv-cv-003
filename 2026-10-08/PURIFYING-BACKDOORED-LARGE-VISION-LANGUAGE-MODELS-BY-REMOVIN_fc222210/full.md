# PURIFYING BACKDOORED LARGE VISION-LANGUAGE MODELS BY REMOVING HIJACKED DIRECTIONS

Bojun Yang<sup>1†</sup> Haochen Zhou<sup>1†</sup> Zhifang Zhang<sup>2</sup> Haobo Wang<sup>3</sup> Songze Li<sup>1</sup> Lei Feng

<sup>1</sup> Southeast University <sup>2</sup> The University of Queensland <sup>3</sup> Zhejiang University

## ABSTRACT

Large vision-language models (LVLMs) are increasingly deployed in safety-critical applications, yet they remain vulnerable to backdoor attacks. Defending against such attacks remains costly, as existing methods require either extensive retraining on clean data or per-query intervention at inference time. To address this limitation, we propose OrthoPurify, a more efficient method to purify backdoored model weights via one-step orthogonal projection. Specifically, through structural analysis of backdoor weight updates, we find that the backdoor is encoded by diverting a small number of weight update directions from task adaptation to backdoor shortcut encoding, a phenomenon we term direction hijacking. However, identifying these hijacked directions requires a benign reference model, which is typically inaccessible to the defender. We show that a pseudo-benign model, obtained by fine-tuning the pretrained weights on only a small set of clean samples, provides a sufficient approximation, as the dominant update directions stabilize within the first few gradient steps. OrthoPurify uses this pseudo-benign reference to isolate the hijacked directions and removes them through a single projection on the weight update. Extensive experiments show that OrthoPurify reduces the attack success rate to near zero while preserving the original performance across diverse benchmarks, without retraining the backdoored model or introducing inference-time overhead. Our code is publicly available at this repository.

## 1 INTRODUCTION

Large vision–language models (LVLMs) (Liu et al., 2024; Dai et al., 2023; Bai et al., 2025) have demonstrated strong multimodal capabilities and are increasingly deployed in safety-critical applications such as medical diagnosis (Guo & Terzopoulos, 2025; Van et al., 2024) and autonomous driving (Zhou et al., 2024; Gao et al., 2025). To adapt to downstream tasks, these models are typically fine-tuned via their lightweight adapter module on task-specific data (Zhu et al., 2024; Li et al., 2023; Ye et al., 2024), a process that is often outsourced to third-party services or performed on data from untrusted sources (Hong et al., 2022; Chen et al., 2020). However, this practice introduces a security risk known as backdoor attack (Ma et al., 2025; Ye et al., 2025; Zhao et al., 2025): by injecting a small number of poisoned samples into the fine-tuning data, an adversary can implant a backdoor (Gu et al., 2019) that causes the model to produce attacker-specified outputs whenever a visual trigger appears at inference time, while behaving normally on clean inputs.

Existing defenses against LVLM backdoors can be broadly categorized into post-training and testtime approaches. Post-training methods (Liu et al., 2018; Min et al., 2025; Wu & Wang, 2021; Wang et al., 2019; Li et al., 2021a) retrain or prune the backdoored model on clean data, but they typically demand hundreds of clean samples, incur substantial computation, and often degrade the original performance. Test-time methods (Li et al., 2021c; Shi et al., 2023; Zhang et al., 2026b; Gao et al., 2019) instead perturb inputs or manipulate internal model signals during inference, avoiding retraining but introducing per-query overhead and leaving the backdoor intact in model weights.

To address these limitations, we propose OrthoPurify, an efficient weight-space purification method that removes backdoor effects through a single orthogonal projection. Rather than intervening on inputs or activations, we turn to the structure of the weight update itself. By comparing the principal subspaces of backdoored and benign weight updates, we find that backdoor fine-tuning does not substantially change the singular-value spectrum of the update. Instead, it diverts a small number of update directions away from task adaptation to the encoding of a backdoor shortcut, while leaving most directions aligned with benign fine-tuning to preserve clean-task performance, a phenomenon we term direction hijacking. As shown in Figure 2(a), when we compare the principal directions of the backdoored and benign weight updates, most are well aligned $( < 3 0 ^ { \circ } )$ , while one direction is nearly orthogonal $( > 7 0 ^ { \circ } )$ , and this pattern is consistent across all six attacks. To verify that this direction encodes the backdoor, we conduct a controlled experiment (Figure 3): removing it from the weight update reduces the attack success rate to 0%, while retaining only this direction partially restores the attack. A natural question is whether existing defenses remove these hijacked directions. To answer this, we measure how much energy along the hijacked directions survives in the weight update after defense. Specifically, we define the hijacked energy retention ratio:

![](images/861455c77d5fc92ec8caaf2395deb0d0f7f8146db034909ded9170c41545dbc2.jpg)  
Figure 1: Overview of OrthoPurify. SVD analysis of the backdoored weight update reveals direction hijacking: most principal directions are shared with a benign reference $( \theta \ : < \ : 3 0 ^ { \circ } )$ , while a few are nearly orthogonal $( \theta > 7 0 ^ { \circ } )$ and encode the backdoor exclusively. A pseudo-benign reference, obtained by fine-tuning pretrained weights on a small clean set for a few gradient steps, provides a sufficient approximation. OrthoPurify identifies the hijacked directions via principal angle thresholding and removes them through a single orthogonal projection.

$$
r = \frac { \sum _ { l } \Vert ( \Delta \mathbf { W } ^ { \prime ( l ) } ) ^ { \top } \mathbf { D } ^ { ( l ) } \Vert _ { F } ^ { 2 } } { \sum _ { l } \Vert ( \Delta \mathbf { W } ^ { ( l ) } ) ^ { \top } \mathbf { D } ^ { ( l ) } \Vert _ { F } ^ { 2 } } ,\tag{1}
$$

where $\Delta { \mathbf W } ^ { ( l ) }$ and $\Delta \mathbf { W } ^ { \prime ( l ) }$ are the weight updates of layer l before and after defense, and $\mathbf { D } ^ { ( l ) }$ collects the hijacked direction vectors of that layer. As shown in Figure 2(b), all baseline defenses retain or amplify the hijacked energy $( r \geq 0 . 9 \dot { 3 } )$ , whereas OrthoPurify reduces it to at most 0.17. Together with their residual ASR in Figure 3, this suggests that existing post-training defenses fail to reliably remove the backdoor at its weight-space source, as they do not directly eliminate the hijacked directions from the weight updates.

Since the hijacked directions are few and well separated from the task-relevant ones, OrthoPurify eliminates the backdoor by directly identifying and projecting them out of the weight update in a single pass. The main practical challenge is that identifying these directions requires comparing the backdoored weight update against a benign reference, which is typically inaccessible to the defender. We show that a pseudo-benign model, obtained by fine-tuning the pretrained weights for just a few gradient steps on a small amount of data, is sufficient to recover the benign directions needed for this comparison. This is because the dominant directions of the weight update stabilize within the first few gradient steps, well before the model converges, as we analyze in detail in Section 3.2.

Building on these findings, the complete method works as follows (Figure 1). Given a backdoored model, OrthoPurify constructs a pseudo-benign reference from a small clean set, compares the two weight updates to identify backdoor-exclusive directions, and removes them through a single projection. The entire procedure requires only a handful of clean samples, with no retraining and no inference-time overhead. Our contributions are:

![](images/fef4009e6fb63b466f137e89a0d651e85af3d6fd268bf742d02cbfecc3f08083.jpg)  
(a) Principal angle analysis

![](images/22b17bd9558500a79a0d0919a532d175156168d1b6456ed79beb9ec57d742f76.jpg)  
(b) Hijacked energy retention ratio  
Figure 2: Direction hijacking in backdoor weights. (a) Principal angles between the backdoor and benign weight update subspaces (k=5) across six attack types on $\mathrm { L L a } \hat { \mathrm { V } } \mathrm { A } { - } 1 . 5 { - } 7 \mathrm { B }$ . Most directions are well aligned $( < 3 0 ^ { \circ } )$ , while one direction consistently exceeds $\mathrm { \dot { 7 } 0 ^ { \circ } }$ (red), indicating direction hijacking. The dashed line marks the $5 0 ^ { \circ }$ threshold used in our method. (b) Hijacked energy retention ratio after applying each defense method. Values below 1.0 indicate removal of hijacked energy; values above 1.0 indicate amplification. All existing defenses retain or amplify the hijacked energy, while OrthoPurify removes most of it.

• A structural characterization of LVLM backdoors in weight space. We identify direction hijacking, a structural signature of LVLM backdoors: backdoor fine-tuning diverts a small number of weight update directions to encode a shortcut, and no existing defense removes them.

• An efficient purification method based on one-step projection. OrthoPurify identifies backdoorexclusive directions by comparing the weight update subspaces of the backdoored model and a pseudo-benign reference, which is constructed by fine-tuning the pretrained weights on a small clean set for a few gradient steps. A single orthogonal projection then removes these directions, requiring no retraining or inference-time overhead.

• Strong empirical results across diverse settings. Across two LVLM architectures (LLaVA-1.5 and Qwen3-VL), six attack types (BadNet, Blended, WaNet, ISSBA, TrojVLM, VLOOD), and multiple fine-tuning paradigms, OrthoPurify reduces the attack success rate to near zero while preserving or improving clean-task performance.

## 2 PRELIMINARY

## 2.1 THREAT MODEL

Victim models. We consider LVLMs that follow the widely adopted encoder-adapter-LLM architecture. A frozen visual encoder ${ \mathcal E } _ { \mathrm { v i s } }$ extracts image features from an input image x, a trainable adapter P then projects these features into the text embedding space, and finally a frozen LLM M generates responses autoregressively. Overall, given an image x and a text query q, the model produces an output sequence o as:

$$
{ \pmb o } = \mathcal { M } ( \mathcal { P } ( \mathcal { E } _ { \mathrm { v i s } } ( { \pmb x } ) ) , { \pmb q } ) .\tag{2}
$$

In practice, adapting an LVLM to a downstream task typically involves fine-tuning only the adapter P on task-specific data, while the encoder and LLM remain fixed. We focus on this adapter-level setting as our primary scenario, consistent with prior work (Lyu et al., 2024; 2025). In Section 4.4, we show that OrthoPurify extends to vision-encoder, mixed, and full-model fine-tuning without modification.

Adversary’s objective. The adversary aims to implant a backdoor into the victim LVLM by poisoning the adapter fine-tuning data. We assume the adversary has full control over the fine-tuning process, including the training data and optimization procedure, but can only modify the adapter parameters P. Since the pretrained checkpoints of the visual encoder and LLM are publicly available (Liu et al., 2024; Bai et al., 2025), any unauthorized modification to these components can be detected by comparing against the released weights. The adversary constructs a poisoned dataset $\mathcal { D } = \mathcal { D } _ { c } \cup \mathcal { D } _ { p } ,$ where $\tilde { \mathcal { D } _ { c } } = \mathsf { \bar { \{ } }  ( { \pmb { x } } , { \pmb { q } } , { \pmb { o } } ) \}$ contains clean samples and $\begin{array} { r } { \bar { \mathcal { D } } _ { p } = \{ ( { \pmb x } \oplus \bar { \Theta } , { \pmb q } , { \pmb o } ^ { * } ) \} } \end{array}$ contains samples with trigger pattern Θ and target output $o ^ { * }$ . After fine-tuning on $\mathcal { D } ,$ the backdoored model $f _ { \theta ^ { * } }$ ∗ behaves normally on clean inputs but produces the target output when the trigger is present:

$$
f _ { \theta ^ { * } } ( x , \pmb { q } ) = \pmb { o } , \quad f _ { \theta ^ { * } } ( \pmb { x } \oplus \Theta , \pmb { q } ) = \pmb { o } ^ { * } .\tag{3}
$$

![](images/1c299af50bcd79e704c52c22de928ec203d781918eed8168d6d5e4ed8064312f.jpg)  
Figure 3: Verification that hijacked directions encode the backdoor. ASR under three conditions on LLaVA 1.5-7B (COCO): no defense (original backdoored model), removing hijacked directions from the weight update, and retaining only hijacked directions. Removing the hijacked directions eliminates the backdoor (ASR → 0%). Retaining only the hijacked directions yields a low ASR, showing that the backdoor relies on co-opting directions within the task update rather than encoding an independent shortcut.

Defender’s setting. The defender receives a fine-tuned model suspected of containing a backdoor and aims to purify it before deployment. We assume the defender only has access to: (i) the pretrained adapter weights $\mathbf { \dot { W } } _ { \mathrm { p r e } } .$ , which are publicly available; (ii) the suspected backdoored adapter weights $\mathbf { W _ { b d } } ;$ and (iii) a small set of clean samples.

## 2.2 SVD AND PRINCIPAL ANGLES

Given the pretrained adapter weights $\mathbf { W } _ { \mathrm { p r e } }$ and a fine-tuned variant W, we define the weight update as $\Delta \mathbf { W } = \mathbf { \bar { W } } - \mathbf { W } _ { \mathrm { p r e } }$ . Its thin SVD is:

$$
\Delta { \bf W } = { \bf U } \Sigma { \bf V } ^ { \top } ,\tag{4}
$$

where U and V are orthonormal matrices of left and right singular vectors, and $\pmb { \Sigma } = \mathrm { d i a g } ( \sigma _ { 1 } , \dots , \sigma _ { r } )$ contains the singular values in decreasing order. We refer to the subspace spanned by the top-k columns of V, the right singular vectors $\{ v _ { 1 } , \ldots , v _ { k } \}$ , as the principal subspace of $\Delta \mathbf { W }$ , denoted S.

To compare two subspaces $\mathcal { S } _ { A }$ and $\boldsymbol { S } _ { B }$ of equal dimension k, we use principal angles (Björck & Golub, 1973). Let $\mathbf { V } _ { A } ^ { \mathbf { \hat { \alpha } } } , \mathbf { V } _ { B } \in \mathbb { R } ^ { n \times k }$ be orthonormal bases for $\mathcal { S } _ { A }$ and $\boldsymbol { S } _ { B }$ . The principal angles $\theta _ { 1 } , \ldots , \theta _ { k } \in [ 0 , \frac { \pi } { 2 } ]$ are defined through the SVD of $\mathbf { V } _ { A } ^ { \top } \mathbf { V } _ { B } { : }$

$$
\begin{array} { r } { \mathbf { V } _ { A } ^ { \top } \mathbf { V } _ { B } = \mathbf { P } \operatorname { d i a g } ( \cos \theta _ { 1 } , \ldots , \cos \theta _ { k } ) \mathbf { Q } ^ { \top } , } \end{array}\tag{5}
$$

where P and Q are orthogonal matrices. A principal angle near 0 indicates that the corresponding directions are shared between the two subspaces, while an angle near $\frac { \pi } { 2 }$ indicates a direction that is present in one subspace but nearly absent from the other.

## 3 OUR METHOD: ORTHOPURIFY

OrthoPurify purifies a backdoored LVLM in a single pass by analyzing the subspace structure of its weight update. Our approach is built on a key observation (Section 3.1): when we compare the principal subspaces of the backdoored and benign weight updates, most directions are well aligned, yet a small number of nearly orthogonal directions consistently emerge that encode the backdoor. Identifying these directions requires a benign reference, which is typically inaccessible; we show that a pseudo-benign model, obtained by fine-tuning the pretrained weights for a few gradient steps (Section 3.2), provides a sufficient approximation, enabling a single projection to remove the backdoor (Section 3.3).

## 3.1 OBSERVATION: DIRECTION HIJACKING IN BACKDOOR WEIGHT UPDATES

To understand the internal structure of backdoor weight updates, we begin by comparing the weight updates produced by backdoor and benign fine-tuning. For this analysis, we assume temporary access to a benign model $\mathbf { W _ { \mathrm { b n } } }$ trained on clean data only; this assumption will be removed in Section 3.2. We compute the weight updates $\Delta \mathbf { W } _ { \mathrm { b d } } = \mathbf { W } _ { \mathrm { b d } } { \bf \dot { \Omega } } - \mathbf { W } _ { \mathrm { p r e } }$ and $\mathbf { \bar { \Delta } } \Delta \mathbf { W } _ { \mathrm { b n } } = \mathbf { W } _ { \mathrm { b n } } - \mathbf { W } _ { \mathrm { p r e } }$ , decompose each via SVD (Eq. 4), and extract their top-k principal subspaces $ { S _ { \mathrm { b d } } }$ and ${ \cal { S } } _ { \mathrm { { b n } } }$ . We then measure the alignment between these two subspaces using the principal angles defined in Eq. 5.

![](images/5a6cd5b44600188e3121691f325afb6539e30fe3d28d7d1626a2e98d7810a79a.jpg)  
(a) Singular value spectra

![](images/ed103b8506a7192d7a7f96bd533352b7f69ba2214515fc1ba19abc67fcd0d1cf.jpg)  
(b) Pseudo-benign direction convergence  
Figure 4: Supporting evidence for direction hijacking and pseudo-benign approximation. (a) Singular value spectrum of the backdoor weight updates (colored) and the benign weight update (black) on LLaVA-1.5-7B. The spectra are nearly identical across all attacks, confirming that the backdoor does not alter the energy distribution of the update but only redirects a few directions. (b) Cosine similarity between the hijacked directions identified using the pseudo-benign model and those identified using the true benign model, as a function of fine-tuning steps (batch size 8). The similarity exceeds 0.97 within 2 steps across all attack types.

The results reveal a consistent pattern across all six attacks (Figure 2(a)). The vast majority of principal angles are small $( < 3 0 ^ { \circ } )$ , indicating that backdoor and benign fine-tuning produce largely aligned weight updates that share most directions for task adaptation. However, a small number of directions exhibit principal angle exceeding $7 0 ^ { \circ }$ , nearly orthogonal to any direction in the benign subspace. We refer to this phenomenon as direction hijacking: rather than introducing additional rank or energy into the weight update, the backdoor redirects a few update directions from task adaptation to encoding a backdoor shortcut. The singular value spectra of $\bar { \Delta \mathbf { W _ { \mathrm { b d } } } }$ and $\Delta \mathbf { W } _ { \mathrm { b n } }$ are nearly identical (Figure 4(a)), confirming that what changes is not how much the weights are updated, but where.

To verify the role of the hijacked directions, we conduct two experiments (Figure 3). First, we remove the contribution of the hijacked directions from $\Delta \mathbf { W } _ { \mathrm { b d } }$ and reconstruct the weights. The attack success rate (ASR) drops to near zero. Second, we retain only the hijacked directions and discard the remaining task-adapted components. The resulting ASR is low, indicating that the backdoor cannot function from these directions alone but depends on the surrounding task-adapted directions to produce the attacker-specified behavior. A controlled experiment comparing hijacked, random, and most-aligned direction removal is provided in Appendix H.

## 3.2 PSEUDO-BENIGN SUBSPACE APPROXIMATION

The analysis in Section 3.1 identifies hijacked directions by comparing against a benign reference model $\mathbf { W } _ { \mathrm { b n } } .$ . In practice, however, the defender only has access to the pretrained weights $\mathbf { W } _ { \mathrm { p r e } }$ and a small set of clean samples. We now show that these are sufficient to construct a pseudo-benign model whose principal subspace closely approximates that of a true benign model.

Construction. Starting from $\mathbf { W } _ { \mathrm { p r e } } .$ , we fine-tune the weights on the small clean set for T gradient steps (typically $T = 2 { \cdot } 1 6 )$ with a standard training objective, producing pseudo-benign weights $\mathbf { W _ { p b } }$ We then compute $\Delta \mathbf { W } _ { \mathrm { p b } } = \mathbf { W } _ { \mathrm { p b } } - \mathbf { W } _ { \mathrm { p r e } }$ and extract its top-k principal subspace $\breve { S _ { \mathrm { p b } } }$ via SVD. We show that $ { S _ { \mathrm { p b } } }$ is a sufficient approximation of ${ \mathcal { S } } _ { \mathrm { b n } }$ for identifying the hijacked directions, even though $\mathbf { W _ { p b } }$ is far from converged and is trained on a much smaller dataset than $\mathbf { W _ { \mathrm { b n } } }$

Intuition. The weight update after T steps of gradient descent can be written as $\Delta { \bf W } _ { T } ~ =$ $\begin{array} { r } { \eta \sum _ { t = 1 } ^ { T } \nabla \mathbf { w } \mathcal { L } _ { t } , } \end{array}$ where $\mathcal { L } _ { t }$ is the loss at step t. The SVD principal directions of $\Delta \mathbf { W } _ { T }$ are determined by the dominant directions along which the per-step gradients $\nabla _ { \mathbf { W } } \mathcal { L } _ { t }$ concentrate, not by the overall magnitude of $\Delta \mathbf { W } _ { T }$ . When the gradients are computed on data from the same underlying distribution, their directional structure is consistent, which means the dominant directions emerge within the first few steps. Additional training steps increase the magnitude of $\Delta \mathbf { W } _ { T }$ but do not substantially change its principal directions. Thus, a few steps of clean fine-tuning are enough to reveal where the benign update would go, even without completing it. We provide a spectral-gap argument supporting this intuition in Appendix N.

Empirical validation. Figure 4(b) shows the cosine similarity between the hijacked directions identified using $ { S _ { \mathrm { p b } } }$ and those identified using the oracle ${ \mathcal { S } } _ { \mathrm { b n } }$ , as a function of the number of finetuning steps. The similarity exceeds 0.97 within just 2 gradient steps and remains stable thereafter, confirming that the pseudo-benign model identifies the same hijacked directions as the oracle benign model.

This result makes the observation in Section 3.1 actionable: the defender can construct a pseudobenign reference from minimal resources, identify the hijacked directions, and proceed to purification without ever having access to a true benign model.

## 3.3 PROJECTION PURIFICATION

With the pseudo-benign subspace $ { S _ { \mathrm { p b } } }$ from Section 3.2 serving as the benign reference, we now describe how to identify the hijacked directions and remove them from the weight updates.

Identifying hijacked directions. Let $\mathbf { V _ { \mathrm { b d } } } , \mathbf { V _ { \mathrm { p b } } } \in \mathbb { R } ^ { n \times k }$ be the orthonormal bases of $S _ { \mathrm { b d } }$ and $ { S _ { \mathrm { p b } } }$ We compute their principal angles via the SVD of $\mathbf { V _ { \mathrm { b d } } ^ { \mathrm { T } } } \mathbf { V _ { \mathrm { p b } } }$

$$
\mathbf { V } _ { \mathrm { b d } } ^ { \top } \mathbf { V } _ { \mathrm { p b } } = \mathbf { P } \mathrm { d i a g } ( \cos \theta _ { 1 } , \ldots , \cos \theta _ { k } ) \mathbf { Q } ^ { \top } .\tag{6}
$$

We treat any direction with a principal angle above a threshold θ as hijacked. Let ${ \mathcal { T } } = \{ i \mid \theta _ { i } > \theta \}$ denote the index set of hijacked directions. The corresponding directions in the original weight space are recovered as $\mathbf { D } = \mathbf { V _ { \mathrm { b d } } } \mathbf { P } _ { : , \mathcal { T } } \in \mathbb { R } ^ { n \times | \mathcal { T } | }$ , where $\mathbf { P } _ { : , \mathcal { Z } }$ denotes the columns of $\mathbf { P }$ indexed by $\mathcal { T } .$ In practice, we set $\theta = 5 0 ^ { \circ }$ . This value falls naturally between the aligned directions $( < 3 0 ^ { \circ } )$ and the hijacked directions $( > 7 0 ^ { \circ } )$ ) observed in Section 3.1, and we show in Section 4 that the method is not sensitive to this choice.

Projection purification. Given the hijacked direction matrix D, we remove their contribution from the backdoor weight update through orthogonal projection:

$$
\mathbf { W } _ { \mathrm { p u r } } = \mathbf { W } _ { \mathrm { b d } } - \Delta \mathbf { W } _ { \mathrm { b d } } \mathbf { D } \mathbf { D } ^ { \top } .\tag{7}
$$

This subtracts the component of $\Delta \mathbf { W } _ { \mathrm { b d } }$ that lies along the hijacked directions while leaving all other components intact. The complete procedure is summarized in Algorithm 1 (Appendix B).

## 4 EXPERIMENTS

## 4.1 EXPERIMENTAL SETTINGS

Victim models and benchmarks. We evaluate on two LVLM architectures with different adapter designs: LLaVA-1.5-7B (Liu et al., 2024), which uses a two-layer MLP as its vision–language projector, and Qwen3-VL-8B (Bai et al., 2025), which adopts a Merger module combined with DeepStack layers. We evaluate on two tasks: image captioning (COCO (Lin et al., 2014)) and visual question answering (VQAv2 (Goyal et al., 2017)).

Backdoor setup. Following prior work (Lyu et al., 2024; 2025; Liang et al., 2025; Zhang et al., 2026b), we fine-tune the adapter on 3,000 image-query-answer pairs using the target phrase "you have been hacked lol" as the adversary’s target output. We evaluate six attack types: BadNet (Gu et al., 2019) (patched trigger), Blended (Chen et al., 2017) (global blending), WaNet (Nguyen & Tran, 2021) (imperceptible warping), ISSBA (Li et al., 2021d) (sample-specific steganography), and two LVLM-specific attacks, TrojVLM (Lyu et al., 2024) and VLOOD (Lyu et al., 2025). For the first four attacks, the target text replaces the original answer entirely; for TrojVLM and VLOOD, it is inserted into the original content. The poison rate for each attack is chosen to ensure sufficient attack strength for a rigorous evaluation of defense effectiveness. Detailed configurations including poison rates are provided in Appendix C.3.

Defense baselines. We compare against four defense methods: Clean Fine-tuning, which continues fine-tuning the backdoored model on clean data; Fine-Pruning (Liu et al., 2018), which prunes dormant neurons and then fine-tunes; ANP (Wu & Wang, 2021), which identifies and prunes backdoorsensitive neurons via adversarial perturbation; and CLP (Zheng et al., 2022), which prunes channels with high spectral norm. Since these methods were originally designed for image classifiers, we adapt them to the LVLM setting and provide sufficient clean data and computation to ensure a fair comparison. Detailed configurations are provided in Appendix C.4.

Evaluation metrics. We report the attack success rate (ASR, ↓), which measures the proportion of triggered inputs that produce the target response, and clean utility (CU, ↑), measured by CIDEr (Vedantam et al., 2015) for image captioning and V-score (Antol et al., 2015) for VQA.

Table 1: Results on image captioning (COCO). ASR (↓%) and CU (CIDEr on clean inputs, ↑) of each defense against six backdoor attacks on LLaVA-1.5-7B and Qwen3-VL-8B.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">BadNet</td><td colspan="2">Blended</td><td colspan="2">WaNet</td><td colspan="2">ISSBA</td><td colspan="2">TrojVLM</td><td colspan="2">VLOOD</td></tr><tr><td>ASR</td><td>CU</td><td>ASR</td><td>CU</td><td>ASR</td><td>CU</td><td>ASR</td><td>CU</td><td>ASR</td><td>CU</td><td>ASR</td><td>CU</td></tr><tr><td rowspan="6">LLaVA-1.5</td><td>No defense</td><td>99.41</td><td>121.48</td><td>97.85</td><td>129.98</td><td>98.63</td><td>121.48</td><td>98.44</td><td>126.74</td><td>100.00</td><td>107.66</td><td>81.64</td><td>117.36</td></tr><tr><td>Clean FT</td><td>98.44</td><td>134.64</td><td>93.16</td><td>137.22</td><td>96.29</td><td>129.83</td><td>83.59</td><td>135.85</td><td>96.09</td><td>128.21</td><td>28.32</td><td>127.60</td></tr><tr><td>Fine-Pruning</td><td>0.59</td><td>129.09</td><td>86.91</td><td>128.90</td><td>52.34</td><td>126.89</td><td>16.99</td><td>129.75</td><td>3.71</td><td>110.11</td><td>0.78</td><td>128.41</td></tr><tr><td>ANP</td><td>74.20</td><td>123.82</td><td>86.00</td><td>121.95</td><td>81.60</td><td>124.54</td><td>8.80</td><td>125.77</td><td>44.80</td><td>120.53</td><td>25.20</td><td>125.02</td></tr><tr><td>CLP</td><td>59.77</td><td>115.82</td><td>51.76</td><td>122.16</td><td>26.17</td><td>120.42</td><td>2.93</td><td>116.11</td><td>50.59</td><td>106.36</td><td>0.98</td><td>120.10</td></tr><tr><td>OrthoPurify</td><td>0</td><td>125.85</td><td>0</td><td>131.79</td><td>0</td><td>126.20</td><td>0</td><td>125.12</td><td>0</td><td>130.08</td><td>0</td><td>121.84</td></tr><tr><td rowspan="6">Qwen3-VL</td><td>No defense</td><td>99.80</td><td>70.92</td><td>95.70</td><td>67.86</td><td>99.80</td><td>68.04</td><td>99.41</td><td>66.01</td><td>100.00</td><td>79.20</td><td>99.80</td><td>24.97</td></tr><tr><td>Clean FT</td><td>58.98</td><td>71.28</td><td>19.34</td><td>49.89</td><td>4.88</td><td>68.27</td><td>36.91</td><td>57.39</td><td>96.09</td><td>62.84</td><td>91.02</td><td>70.44</td></tr><tr><td>Fine-Pruning</td><td>96.88</td><td>69.73</td><td>55.08</td><td>55.68</td><td>1.17</td><td>64.13</td><td>16.02</td><td>56.18</td><td>70.31</td><td>66.80</td><td>22.66</td><td>63.06</td></tr><tr><td>ANP</td><td>99.80</td><td>71.08</td><td>95.70</td><td>71.42</td><td>99.80</td><td>76.92</td><td>99.41</td><td>65.74</td><td>100.00</td><td>32.45</td><td>99.80</td><td>25.12</td></tr><tr><td>CLP</td><td>100.00</td><td>71.23</td><td>96.29</td><td>57.27</td><td>99.22</td><td>55.54</td><td>93.95</td><td>72.14</td><td>99.80</td><td>38.53</td><td>96.29</td><td>17.14</td></tr><tr><td>OrthoPurify</td><td>0</td><td>71.59</td><td>0</td><td>69.95</td><td>0</td><td>69.79</td><td>0</td><td>64.94</td><td>0</td><td>74.06</td><td>0</td><td>25.03</td></tr></table>

Table 2: Results on VQA (VQAv2). ASR (↓%) and CU (V-score on clean inputs, ↑%) of each defense against six backdoor attacks on LLaVA-1.5-7B and Qwen3-VL-8B.
<table><tr><td rowspan="2">Model</td><td rowspan="2">Method</td><td colspan="2">BadNet</td><td colspan="2">Blended</td><td colspan="2">WaNet</td><td colspan="2">ISSBA</td><td colspan="2"> $\mathrm { T r o j V L M }$ </td><td colspan="2">VLOOD</td></tr><tr><td>ASR</td><td>CU</td><td>ASR</td><td>CU</td><td>ASR</td><td>CU</td><td>ASR</td><td>CU</td><td>ASR</td><td>CU</td><td>ASR</td><td>CU</td></tr><tr><td rowspan="6">LLaVA-1.5</td><td>No defense</td><td>92.29</td><td>76.92</td><td>98.83</td><td>67.84</td><td>95.70</td><td>76.14</td><td>99.22</td><td>66.41</td><td>99.80</td><td>75.42</td><td>99.41</td><td>70.18</td></tr><tr><td>Clean FT</td><td>99.22</td><td>67.06</td><td>98.05</td><td>69.27</td><td>87.50</td><td>69.14</td><td>94.34</td><td>64.52</td><td>99.02</td><td>65.43</td><td>95.51</td><td>69.14</td></tr><tr><td>Fine-Pruning</td><td>4.49</td><td>65.36</td><td>97.85</td><td>64.71</td><td>21.68</td><td>67.58</td><td>89.45</td><td>63.61</td><td>69.92</td><td>76.63</td><td>0</td><td>64.32</td></tr><tr><td>ANP</td><td>83.01</td><td>62.96</td><td>90.43</td><td>53.52</td><td>4.10</td><td>36.46</td><td>0.20</td><td>17.97</td><td>42.38</td><td>52.02</td><td>99.02</td><td>0.13</td></tr><tr><td>CLP</td><td>4.10</td><td>71.16</td><td>94.34</td><td>56.71</td><td>19.34</td><td>71.48</td><td>92.19</td><td>60.55</td><td>4.30</td><td>69.01</td><td>84.77</td><td>62.43</td></tr><tr><td>OrthoPurify</td><td>0</td><td>77.47</td><td>0</td><td>65.49</td><td>0</td><td>79.69</td><td>0</td><td>66.80</td><td>0</td><td>76.69</td><td>0</td><td>61.39</td></tr><tr><td rowspan="6">Qwen3-VL</td><td>No defense</td><td>100.00</td><td>84.24</td><td>97.46</td><td>83.56</td><td>96.78</td><td>84.86</td><td>99.80</td><td>83.07</td><td>100.00</td><td>83.72</td><td>100.00</td><td>85.29</td></tr><tr><td>Clean FT</td><td>7.23</td><td>58.01</td><td>0.20</td><td>8.66</td><td>0</td><td>36.65</td><td>0</td><td>64.97</td><td>0</td><td>37.30</td><td>93.95</td><td>85.68</td></tr><tr><td>Fine-Pruning</td><td>94.34</td><td>87.83</td><td>62.50</td><td>85.68</td><td>54.88</td><td>86.00</td><td>0</td><td>84.11</td><td>99.80</td><td>86.52</td><td>0</td><td>84.37</td></tr><tr><td>ANP</td><td>100.00</td><td>84.64</td><td>96.88</td><td>84.83</td><td>94.92</td><td>86.20</td><td>99.80</td><td>84.51</td><td>100.00</td><td>84.11</td><td>100.00</td><td>85.29</td></tr><tr><td>CLP</td><td>100.00</td><td>84.64</td><td>93.36</td><td>84.38</td><td>94.73</td><td>85.68</td><td>99.61</td><td>83.79</td><td>98.24</td><td>80.36</td><td>100.00</td><td>84.51</td></tr><tr><td>OrthoPurify</td><td>0</td><td>79.95</td><td>0</td><td>86.00</td><td>0</td><td>84.57</td><td>0</td><td>84.77</td><td>0</td><td>81.25</td><td>0</td><td>84.57</td></tr></table>

Implementation details. For our method, we apply the purification procedure to all 2D weight matrices in the adapter module, with subspace dimension k = 10 and angle threshold $\theta = 5 0 ^ { \circ }$ . The pseudo-benign model is obtained by fine-tuning for $T = 8$ gradient steps with a batch size of 16, using a total of 64 clean samples. All experiments are conducted on NVIDIA RTX 3090 GPUs.

## 4.2 MAIN RESULTS

Tables 1 and 2 report results across two models, two tasks, and six attacks. OrthoPurify reduces ASR to 0% across all settings, confirming that removing the hijacked directions is sufficient to completely eliminate the backdoor.

No baseline achieves comparable reliability. Clean Fine-tuning either fails to reduce ASR on most attacks or achieves reduction at severe CU cost. Specifically, Clean Fine-tuning leaves ASR above 83% for five of six attacks on LLaVA COCO; on Qwen3-VL VQA, it lowers Blended ASR to 0.20% but collapses CU from 83.56 to 8.66, suggesting that it overwrites learned representations rather than selectively removing the backdoor. Fine-Pruning is also unpredictable. On LLaVA COCO, it reduces BadNet ASR to 0.59% but leaves Blended at 86.91%, and which attacks it handles varies across models without a consistent pattern. ANP and CLP are unreliable on LLaVA and fail entirely on Qwen3-VL, where both leave ASR above 93% on every attack. Even when ANP achieves low ASR, CU is often severely degraded. For example, on LLaVA VQA, it lowers ISSBA to 0.20% while CU drops from 66.41 to 17.97. In contrast, OrthoPurify reliably removes the backdoor across all settings.

OrthoPurify preserves or improves clean utility in most settings. On COCO, removing the hijacked directions often restores CU that backdoor fine-tuning had degraded (e.g., TrojVLM on LLaVA: 107.66 → 130.08). For VLOOD on Qwen3-VL (COCO), the attack’s auxiliary training losses independently degrade CU to 24.97. Our method matches this level (25.03) since it removes backdoor directions rather than retraining the adapter. The consistent effectiveness across both architectures, despite their different adapter designs, suggests that direction hijacking is a structural property of fine-tuning-based backdoors rather than an artifact of a specific model. In contrast to retraining-based defenses that modify the weights broadly, OrthoPurify targets only the hijacked directions and preserves task-adapted components. This directional precision explains the consistent CU preservation across both architectures and tasks. A detailed analysis of per-setting CU variations is provided in Appendix G.

![](images/5b0ed65dafa0e4577cddd8e0e51e583e4da552b64d96f678f5e73b0d58a3023d.jpg)  
Figure 5: Ablation study. Top row: ASR (↓); bottom row: CIDEr (↑). (a) Effect of total clean sample count (batch ${ \mathrm { s i z e } } = n / 8 ,$ , 1 epoch). (b) Effect of fine-tuning steps (batch size 8 fixed). (c) Effect of subspace dimension $k ( \theta { = } 5 0 ^ { \circ }$ fixed). (d) Effect of angle threshold θ (k=10 fixed). All experiments use LLaVA-1.5-7B on COCO under four attack types.

## 4.3 ABLATION STUDIES

We ablate four hyperparameters on LLaVA-1.5-7B (COCO) under four attacks (Figure 5). The method is robust to clean sample count (ASR = 0% with as few as 8 samples) and angle threshold $\theta ( \mathrm { A S R } = 0 \%$ across [30<sup>◦</sup>, 70<sup>◦</sup>], consistent with the angular separation in Section 3.1). ASR drops to 0% from the first gradient step, though $T { \geq } 2$ is needed for CU to stabilize, consistent with the direction convergence in Figure 4(b). The subspace dimension k is the most influential parameter. At $k { = } 3$ , several attacks retain high ASR, but ASR reaches 0% for all attacks at $k { \geq } 1 0$ . Overall, with $k { \geq } 1 0$ and $T { \geq } 2 ,$ both ASR and CU are stable across all settings in Figure 5.

## 4.4 FURTHER ANALYSIS

Generalization across fine-tuning paradigms. Our main experiments assume adapter-level backdoors. To test whether OrthoPurify generalizes beyond this setting, we evaluate on three fine-tuning scopes: adapter only, mixed (vision encoder + adapter, 324M parameters), and full model (7.06B parameters), as shown in Table 3. Extending the fine-tuning scope to mixed or full-model tuning does not degrade purification effectiveness, as the method applies the same per-matrix SVD analysis regardless of which parameters were fine-tuned. We further test against BadVision (Liu & Zhang, $2 0 \hat { 2 } 5 )$ , where only the CLIP vision encoder is backdoored with the adapter and LLM frozen. OrthoPurify reduces ASR from 96.88% to 6.84%, close to the clean false-alarm rate of 2.73%, while CIDEr improves slightly (121.09 → 123.12). These results confirm that direction hijacking is a general property of fine-tuning-based backdoors beyond adapter training.

Robustness against adaptive attacks. We evaluate an adaptive attacker who knows the defense mechanism and optimizes the backdoor training to evade detection. The attacker adds an alignment regularizer to the training loss:

$$
\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { C E } } + \lambda \left. \Delta \mathbf { W } \left( \mathbf { I } - \mathbf { V } _ { \mathrm { p b } } \mathbf { V } _ { \mathrm { p b } } ^ { \top } \right) \right. _ { F } ^ { 2 } , } \end{array}
$$

which penalizes weight-update energy outside the pseudo-benign subspace, reducing the principal angles that OrthoPurify relies on. We sweep λ on BadNet (LLaVA-1.5-7B, COCO). Against this attack, the defender lowers θ from the default $5 0 ^ { \circ }$ to 20<sup>◦</sup>.

As shown in Figure 6, OrthoPurify at $\theta { = } 2 0 ^ { \circ }$ reduces ASR to 0% across all λ values with CU comparable to the undefended model. The attacker faces an unavoidable trade-off: for $\lambda \le 0 . 2 7$ , the backdoor remains functional $( \mathrm { A S R } \ge 9 3 \% )$ but OrthoPurify eliminates it entirely; for $\lambda \geq 0 . 2 8$ , the alignment constraint becomes so restrictive that the backdoor collapses on its own $( \mathrm { A S R } \le 5 . 5 \%$ without any defense). While the regularizer compresses the principal angles, it cannot fully eliminate the angular deviation. In practice, the defender can simply adopt the lowest θ that preserves CU, which suffices to neutralize even adaptive attackers. An energy-dispersal attack and multi-θ analysis are provided in Appendix K.

Table 3: Results under different fine-tuning scopes. ASR (↓%) and CU (CIDEr on clean inputs, ↑) with and without OrthoPurify, where the backdoor is injected by fine-tuning the adapter only (Adapter), the vision encoder and adapter (Mixed), or the full model (Full).
<table><tr><td colspan="3"></td><td colspan="2">BadNet</td><td colspan="2">TrojVLM</td></tr><tr><td>Scope</td><td>Model</td><td>Method</td><td>ASR</td><td>CU</td><td>ASR</td><td>CU</td></tr><tr><td rowspan="6">Adapter</td><td rowspan="2">LLaVA-7B (LoRA)</td><td>No defense</td><td>100</td><td>124.37</td><td>100</td><td>124.49</td></tr><tr><td>OrthoPurify</td><td>0</td><td>123.52</td><td>1.56</td><td>124.36</td></tr><tr><td rowspan="2">LLaVA-13B (Adapter)</td><td>No defense</td><td>100</td><td>129.47</td><td>100</td><td>127.36</td></tr><tr><td>OrthoPurify</td><td>0</td><td>131.19</td><td>0</td><td>134.59</td></tr><tr><td rowspan="2">Qwen3-VL-4B (Adapter)</td><td>No defense</td><td>100</td><td>62.57</td><td>99.80</td><td>60.60</td></tr><tr><td>OrthoPurify</td><td>0</td><td>55.01</td><td>0</td><td>58.41</td></tr><tr><td rowspan="2">Mixed</td><td rowspan="2">LLaVA-7B (324M)</td><td>No defense</td><td>100</td><td>125.73</td><td>100</td><td>122.87</td></tr><tr><td>OrthoPurify</td><td>0</td><td>131.45</td><td>0</td><td>120.79</td></tr><tr><td rowspan="2">Full</td><td rowspan="2">LLaVA-7B (7.06B)</td><td>No defense</td><td>100</td><td>131.23</td><td>100</td><td>128.45</td></tr><tr><td>OrthoPurify</td><td>0</td><td>133.40</td><td>0</td><td>134.47</td></tr></table>

Comparison with test-time defenses. We compare with two recent test-time LVLM backdoor defenses: PurMM (Jiang et al., 2026) and Clean-Sight (Zhang et al., 2026b). Since these methods intervene per query during inference rather than purifying weights before deployment, they belong to a different defense paradigm and are reported separately from the main results. On BadNet, all three methods achieve 0% ASR, but on TrojVLM, PurMM and CleanSight retain 50.0% and 11.7% ASR respectively, while OrthoPurify achieves 0%. TrojVLM inserts the target phrase at a random position within the original caption, producing a subtler anomaly that per-query methods struggle to detect. OrthoPurify avoids this limitation by operating in weight space. Furthermore, OrthoPurify is a one-time offline operation with zero inference overhead, becoming more efficient than PurMM after 16 queries and CleanSight after 68 queries. Full details are in Appendix I.

![](images/655f106cde3641eb5133932219ad422d9079f2b553031139cc1b9178f4fe6bad.jpg)  
Figure 6: Adaptive attack via alignment regularization. Solid: ASR; dashed: CU. The attacker faces an unavoidable trade-off at λ ≈ 0.28.

Benign model safety. OrthoPurify can be applied as a precautionary measure without knowing whether the model is backdoored. When applied to benign fine-tuned models (poison rate = 0) across both architectures and both tasks, CU is largely preserved (e.g., LLaVA CIDEr 125.23 → 133.07; Qwen3-VL VQA Score 84.24 → 85.87; Appendix J). If the model turns out to be clean, CU remain stable; if it is backdoored, the threat is removed. This makes OrthoPurify suitable for routine deployment in model supply chains where the backdoor status is unknown a priori.

Additional robustness evaluations. We further verify OrthoPurify under clean-data mismatch (three data sources including cross-task), black-box transferability (surrogate-model attacks), multi-target attacks, and varying poison rates, all with consistent ASR reduction to near zero. We also extend to image classification on ImageNet with CLIP ViT-L/14, where ASR drops from 95.29% to 0% with no accuracy loss, confirming the generality of direction hijacking beyond captioning. Detailed results for each setting are provided in Appendix L.

## 5 CONCLUSION

We proposed OrthoPurify, a one-step weight purification method for defending LVLMs against backdoor attacks. Our analysis revealed that backdoor fine-tuning encodes the backdoor by hijacking a small number of weight update directions, a phenomenon we term direction hijacking. We showed that a pseudo-benign model constructed from a small set of clean samples is sufficient to identify these directions, and a single orthogonal projection removes them without retraining or inference-time overhead. Experiments showed that OrthoPurify reduces ASR to near zero in all settings while preserving clean utility. The consistent effectiveness across distinct architectures and fine-tuning paradigms suggests that direction hijacking is a general structural property of fine-tuning-based backdoors, and that projecting them out of the weight update is a viable defense for LVLMs.

## ETHICS STATEMENT

This work aims to improve the safety of large vision-language models by defending against backdoor attacks. While our structural analysis of backdoor weight updates could theoretically inform adversaries, all attack methods studied are already publicly available, and our contribution is purely defensive.

## REPRODUCIBILITY STATEMENT

We provide full implementation details in Appendix C, including model configurations (Appendix C.1), dataset descriptions (Appendix C.2), backdoor attack implementations (Appendix C.3), and defense baseline settings (Appendix C.4). Our code is publicly available at this repository.

## REFERENCES

Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh. VQA: Visual question answering. In ICCV, 2015.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-VL technical report. arXiv preprint arXiv:2511.21631, 2025.

Mauro Barni, Kassem Kallas, and Benedetta Tondi. A new backdoor attack in CNNS by training set corruption without label poisoning. In ICIP, 2019.

Åke Björck and Gene H. Golub. Numerical methods for computing angles between linear subspaces. Mathematics ofComputation, 1973.

Bryant Chen, Wilka Carvalho, Nathalie Baracaldo, Heiko Ludwig, Benjamin Edwards, Taesung Lee, Ian Molloy, and Biplav Srivastava. Detecting backdoor attacks on deep neural networks by activation clustering. arXiv preprint arXiv:1811.03728, 2018.

Xinyun Chen, Chang Liu, Bo Li, Kimberly Lu, and Dawn Song. Targeted backdoor attacks on deep learning systems using data poisoning. arXiv preprint arXiv:1712.05526, 2017.

Yanjiao Chen, Xueluan Gong, Qian Wang, Xing Di, and Huayang Huang. Backdoor attacks and defenses for deep neural networks in outsourced cloud environments. IEEE Network, 2020.

Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi. InstructBLIP: Towards general-purpose vision-language models with instruction tuning. In NeurIPS, 2023.

Haoxiang Gao, Li Zhang, Yu Zhao, Zhou Yang, and Jinghan Cao. Application of vision-language models to pedestrian behavior prediction and scene understanding in autonomous driving. In RAIIC, 2025.

Yansong Gao, Chang Xu, Derui Wang, Shiping Chen, Damith C. Ranasinghe, and Surya Nepal. STRIP: a defence against trojan attacks on deep neural networks. In ACSAC, 2019.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017.

Tianyu Gu, Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. BadNets: Evaluating backdooring attacks on deep neural networks. IEEE Access, 2019.

Danfeng Guo and Demetri Terzopoulos. Prompting medical large vision-language models to diagnose pathologies by visual question answering. Machine Learningfor Biomedical Imaging, 2025.

Shuo He, Zhifang Zhang, Feng Liu, Roy Ka-Wei Lee, Bo An, and Lei Feng. A closer look at backdoor attacks on CLIP. In ICML, 2025.

Sanghyun Hong, Nicholas Carlini, and Alexey Kurakin. Handcrafted backdoors in deep neural networks. In NeurIPS, 2022.

Wenzheng Jiang, Ke Liang, Xuankun Rong, Jingxuan Zhou, Zhengyi Zhong, Guancheng Wan, and Ji Wang. PurMM: Attention-guided test-time backdoor purification in multimodal large language models. In AAAI, 2026.

Chunyuan Li, Cliff Wong, Sheng Zhang, Naoto Usuyama, Haotian Liu, Jianwei Yang, Tristan Naumann, Hoifung Poon, and Jianfeng Gao. LLaVA-Med: Training a large language-and-vision assistant for biomedicine in one day. In NeurIPS, 2023.

Yige Li, Xixiang Lyu, Nodens Koren, Lingjuan Lyu, Bo Li, and Xingjun Ma. Anti-backdoor learning: Training clean models on poisoned data. In NeurIPS, 2021a.

Yige Li, Xixiang Lyu, Nodens Koren, Lingjuan Lyu, Bo Li, and Xingjun Ma. Neural attention distillation: Erasing backdoor triggers from deep neural networks. In ICLR, 2021b.

Yiming Li, Tongqing Zhai, Baoyuan Wu, Yong Jiang, Zhifeng Li, and Shutao Xia. Rethinking the trigger of backdoor attack. arXiv preprint arXiv:2004.04692, 2021c.

Yuezun Li, Yiming Li, Baoyuan Wu, Longkang Li, Ran He, and Siwei Lyu. Invisible backdoor attack with sample-specific triggers. In ICCV, 2021d.

Siyuan Liang, Jiawei Liang, Tianyu Pang, Chao Du, Aishan Liu, Mingli Zhu, Xiaochun Cao, and Dacheng Tao. Revisiting backdoor attacks against large vision-language models from domain shift. In CVPR, 2025.

Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C. Lawrence Zitnick. Microsoft COCO: common objects in context. In ECCV, 2014.

Weilin Lin, Li Liu, Shaokui Wei, Jianze Li, and Hui Xiong. Unveiling and mitigating backdoor vulnerabilities based on unlearning weight changes and backdoor activeness. In NeurIPS, 2024.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In CVPR, 2024.

Kang Liu, Brendan Dolan-Gavitt, and Siddharth Garg. Fine-Pruning: Defending against backdooring attacks on deep neural networks. In RAID, 2018.

Zhaoyi Liu and Huan Zhang. Stealthy backdoor attack in self-supervised learning vision encoders for large vision language models. In CVPR, 2025.

Weimin Lyu, Lu Pang, Tengfei Ma, Haibin Ling, and Chao Chen. TrojVLM: Backdoor attack against vision language models. In ECCV, 2024.

Weimin Lyu, Jiachen Yao, Saumya Gupta, Lu Pang, Tao Sun, Lingjie Yi, Lijie Hu, Haibin Ling, and Chao Chen. Backdooring vision-language models with out-of-distribution data. In ICLR, 2025.

Xingjun Ma, Yifeng Gao, Yixu Wang, Ruofan Wang, Xin Wang, Ye Sun, Yifan Ding, Hengyuan Xu, Yunhao Chen, Yunhan Zhao, Hanxun Huang, Yige Li, Yutao Wu, Jiaming Zhang, Xiang Zheng, Yang Bai, Yiming Li, Zuxuan Wu, Xipeng Qiu, Jingfeng Zhang, Xudong Han, Haonan Li, Jun Sun, Cong Wang, Jindong Gu, Baoyuan Wu, Siheng Chen, Tianwei Zhang, Yang Liu, Mingming Gong, Tongliang Liu, Shirui Pan, Cihang Xie, Tianyu Pang, Yinpeng Dong, Ruoxi Jia, Yang Zhang, Shiqing Ma, Xiangyu Zhang, Neil Gong, Chaowei Xiao, Sarah Erfani, Tim Baldwin, Bo Li, Masashi Sugiyama, Dacheng Tao, James Bailey, and Yu-Gang Jiang. Safety at scale: a comprehensive survey of large model and agent safety. Foundations and Trends in Privacy and Security, 2025.

Nay Myat Min, Long H. Pham, and Jun Sun. Unified neural backdoor removal with only few clean samples through unlearning and relearning. T-IFS, 2025.

Tuan Anh Nguyen and Anh Tran. Input-aware dynamic backdoor attack. In NeurIPS, 2020.

Tuan Anh Nguyen and Anh Tuan Tran. WaNet - imperceptible warping-based backdoor attack. In ICLR, 2021.

Xuankun Rong, Wenke Huang, Jian Liang, Jinhe Bi, Xun Xiao, Yiming Li, Bo Du, and Mang Ye. Backdoor cleaning without external guidance in MLLM fine-tuning. In NeurIPS, 2025.

Yucheng Shi, Mengnan Du, Xuansheng Wu, Zihan Guan, Jin Sun, and Ninghao Liu. Black-box backdoor defense via zero-shot image purification. In NeurIPS, 2023.

Brandon Tran, Jerry Li, and Aleksander Madry. Spectral signatures in backdoor attacks. In NeurIPS, 2018.

Joel A. Tropp. User-friendly tail bounds for sums of random matrices. Found. Comput. Math., 2012.

Minh–Hao Van, Prateek Verma, and Xintao Wu. On large visual language models for medical imaging analysis: An empirical study. In CHASE, 2024.

Ramakrishna Vedantam, C. Lawrence Zitnick, and Devi Parikh. CIDEr: Consensus-based image description evaluation. In CVPR, 2015.

Bolun Wang, Yuanshun Yao, Shawn Shan, Huiying Li, Bimal Viswanath, Haitao Zheng, and Ben Y. Zhao. Neural Cleanse: Identifying and mitigating backdoor attacks in neural networks. In IEEE S&P, 2019.

Xinyi Wang, Zhiyu Zhu, Zhibo Jin, Huaming Chen, and Teng Joon Lim. Rethinking Lipschitzness data-free backdoor defense. In CIKM, 2025.

Per-Åke Wedin. Perturbation bounds in connection with singular value decomposition. BIT, 1972.

Dongxian Wu and Yisen Wang. Adversarial neuron pruning purifies backdoored deep models. In NeurIPS, 2021.

Shuhan Xu, Siyuan Liang, Hongling Zheng, Aishan Liu, Xinbiao Wang, Yong Luo, Fu Lin, Leszek Rutkowski, and Dacheng Tao. SRD: Reinforcement-learned semantic perturbation for backdoor defense in VLMs. arXiv preprint arXiv:2506.04743, 2025.

Mang Ye, Xuankun Rong, Wenke Huang, Bo Du, Nenghai Yu, and Dacheng Tao. A survey of safety on large vision-language models: Attacks, defenses and evaluations. arXiv preprint arXiv:2502.14881, 2025.

Qinghao Ye, Haiyang Xu, Jiabo Ye, Ming Yan, Anwen Hu, Haowei Liu, Qi Qian, Ji Zhang, and Fei Huang. mPLUG-Owl2: Revolutionizing multi-modal large language model with modality collaboration. In CVPR, 2024.

Yi Zeng, Si Chen, Won Park, Zhuoqing Mao, Ming Jin, and Ruoxi Jia. Adversarial unlearning of backdoors via implicit hypergradient. In ICLR, 2022.

Zhifang Zhang, Shuo He, Haobo Wang, Bingquan Shen, and Lei Feng. Defending multimodal backdoored models by repulsive visual prompt tuning. In NeurIPS, 2025.

Zhifang Zhang, Qiqi Tao, Jiaqi Lv, Na Zhao, Lei Feng, and Joey Tianyi Zhou. TokenSwap: Backdoor attack on the compositional understanding of large vision-language models. In ICML, 2026a.

Zhifang Zhang, Bojun Yang, Shuo He, Weitong Chen, Wei Emma Zhang, Olaf Maennel, Lei Feng, and Miao Xu. Test-time attention purification for backdoored large vision language models. In CVPR, 2026b.

Zhifang Zhang, Jiahan Zhang, Shengjie Zhou, Qi Wei, Shuo He, Feng Liu, and Lei Feng. Improving generalizability and undetectability for targeted adversarial attacks on multimodal pre-trained models. TPAMI, 2026c.

Pinlong Zhao, Weiyao Zhu, Pengfei Jiao, Di Gao, and Ou Wu. Data poisoning in deep learning: A survey. arXiv preprint arXiv:2503.22759, 2025.

Runkai Zheng, Rongjun Tang, Jianze Li, and Li Liu. Data-free backdoor removal based on channel Lipschitzness. In ECCV, 2022.

Xingcheng Zhou, Mingyu Liu, Ekim Yurtsever, Bare Luka Zagar, Walter Zimmer, Hu Cao, and Alois C. Knoll. Vision language models in autonomous driving: A survey and outlook. T-IV, 2024.

Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. In ICLR, 2024.

# Purifying Backdoored Large Vision-Language Models by Removing Hijacked Directions Appendix

We summarize the Appendix as follows:

• Appendix A provides a discussion of related work.

• Appendix B provides the complete algorithm.

• Appendix C provides detailed experimental settings, including model configurations (Appendix C.1), information on the datasets (Appendix C.2), backdoor attack implementations (Appendix C.3), and defense baseline implementations (Appendix C.4).

• Appendix D provides a detailed efficiency comparison across defense methods.

• Appendix E analyzes resistance to backdoor reactivation after defense.

• Appendix F shows qualitative examples of model outputs before and after purification.

• Appendix G discusses special cases in the main experiment and provides an explanation.

• Appendix H provides a controlled experiment verifying that direction identity determines purification effectiveness.

• Appendix I compares OrthoPurify with two recent test-time LVLM backdoor defenses.

• Appendix J evaluates OrthoPurify on benign (clean) fine-tuned models.

• Appendix K provides a detailed analysis of two adaptive attack strategies targeting OrthoPurify.

• Appendix L provides additional robustness evaluations under non-standard conditions.

• Appendix M provides practical hyperparameter selection guidelines.

• Appendix N provides a formal justification for the subspace convergence property of the pseudobenign approximation.

## A RELATED WORK

Backdoor attacks on vision-language models. Backdoor attacks have emerged as a growing security concern for LVLMs (Ma et al., 2025; Ye et al., 2025; Zhang et al., 2026a;c). The trigger designs used in these attacks have grown increasingly stealthy, progressing from localized patches (Gu et al., 2019) and global image blending (Chen et al., 2017) to imperceptible designs such as spatial warping, sinusoidal signals, and sample-specific encoding (Nguyen & Tran, 2021; Barni et al., 2019; Nguyen & Tran, 2020; Li et al., 2021d). While originally proposed for image classifiers, these methods have been shown to transfer effectively to LVLM fine-tuning (Liang et al., 2025). More recent work has introduced attacks specifically designed for LVLMs. TrojVLM (Lyu et al., 2024) adds a semantic-preserving loss to maintain caption quality, and VLOOD (Lyu et al., 2025) shows that backdoors can be implanted even with out-of-distribution data.

Backdoor defenses on vision-language models. Most existing backdoor defenses have been developed for image classifiers. Post-training methods (Liu et al., 2018; Wu & Wang, 2021; Li et al., 2021b;a; Zeng et al., 2022; Min et al., 2025) retrain or prune the model on clean data to suppress backdoor behavior, but they typically require hundreds of clean samples and often degrade clean-task performance. Test-time methods (Gao et al., 2019; Chen et al., 2018) detect or filter poisoned inputs at inference without modifying the model, but they introduce per-query overhead and leave the backdoor in the weights. More recently, several defenses have been proposed specifically for LVLMs (Rong et al., 2025; Jiang et al., 2026; Xu et al., 2025; Zhang et al., 2026b; 2025; He et al., 2025), addressing backdoor threats through fine-tuning-time cleaning, attention analysis, or input perturbation. Despite this progress, all existing methods either require iterative retraining or intervene at every inference step. In contrast, our method requires neither of these processes, purifying the adapter weights in a single matrix projection.

Structural analysis of weight updates. A number of works have examined the structure of model weights or representations to understand or mitigate backdoors. Spectral Signatures (Tran et al., 2018) applies SVD to hidden-representation covariance to identify poisoned samples, but targets activations rather than weights. CLP (Zheng et al., 2022) and LPP (Wang et al., 2025) prune backdoorsensitive channels based on spectral norms, and TSBD (Lin et al., 2024) locates backdoor neurons by monitoring weight changes during unlearning. These methods all operate at the channel or neuron level, reducing weights to scalar scores rather than analyzing the subspace structure of the full update.

Our work operates at the subspace level: we decompose the adapter weight update via SVD and use principal angle analysis to isolate directions that are exclusive to the backdoor.

## B ALGORITHM

Algorithm 1 OrthoPurify: One-Step Projection for Backdoor Weight Purification   
Require: Pretrained adapter weights $\mathbf { W } _ { \mathrm { p r e } } ,$ suspected backdoored weights $\mathbf { W _ { b d } } .$ , small clean dataset   
$\mathcal { D } _ { \mathrm { c l e a n } } ,$ subspace dimension $\check { k , }$ angle threshold $\theta ,$ fine-tuning steps $\mathbf { \bar { \rho } } _ { T }$   
Ensure: Purified adapter weights $\mathbf { W } _ { \mathrm { p u r } }$   
1: // Backdoor subspace extraction   
2: $\Delta \mathbf { W } _ { \mathrm { b d } }  \mathbf { W } _ { \mathrm { b d } }  \mathbf { W } _ { \mathrm { p r e } }$   
3: $\mathbf { V _ { \mathrm { b d } } }  \mathrm { t o p } { - } k$ right singular vectors of $\Delta \mathbf { W } _ { \mathrm { b d } }$   
4: // Pseudo-benign subspace construction (Sec. 3.2)   
5: $\mathbf { W _ { p b } } $ FineTune $( \mathbf { W } _ { \mathrm { p r e } } , \mathcal { D } _ { \mathrm { c l e a n } } , T )$   
6: $\Delta \dot { \mathbf { W } } _ { \mathrm { p b } }  \mathbf { W } _ { \mathrm { p b } } - \mathbf { W } _ { \mathrm { f } }$ pre   
7: $\mathbf { V } _ { \mathrm { p b } } \dot {  } \mathrm { t o p } – k$ right singular vectors of $\Delta \mathbf { W } _ { \mathrm { p b } }$   
8: // Hijacked direction identification $( S e c . \ 3 . I )$   
9: $\mathbf { V } _ { \mathrm { b d } } ^ { \top } \mathbf { V } _ { \mathrm { p b } } = \mathbf { P }$ diag(cos $\boldsymbol { \theta } _ { 1 } , \ldots , \cos \theta _ { k } ) \mathbf { Q } ^ { \top } \quad \triangleright$ Principal angles   
10: $\mathcal { T } \overset { \right. } { \left. } \left\{ i \mid \theta _ { i } > \theta \right\}$ ▷ Angle thresholding   
11: $\mathbf { D } \gets \mathbf { \dot { V } _ { \mathrm { b d } } } \mathbf { P } [ : , \mathcal { T } ]$ ▷ Hijacked directions   
12: // Projection purification $( S e c . 3 . 3 )$   
13: $\mathbf { W } _ { \mathrm { p u r } }  \mathbf { W } _ { \mathrm { b d } } - \Delta \mathbf { W } _ { \mathrm { b d } } \mathbf { D } \mathbf { D } ^ { \top }$   
14: return $\mathbf { W _ { \mathrm { p u r } } }$

## C DETAILED EXPERIMENTAL SETTINGS

## C.1 MODELS

• LLaVA-1.5-7B (Liu et al., 2024) combines a CLIP ViT-L/336 visual encoder with the Vicuna-7B LLM, using a two-layer MLP projector that maps 576 visual tokens into the LLM’s embedding space. LLaVA-1.5 is widely adopted in LVLM backdoor research (Lyu et al., 2024; 2025; Liang et al., 2025), making it a standard benchmark for fair comparison. In our setting, only the projector is fine-tuned on downstream data, while the encoder and LLM remain frozen.

• Qwen3-VL-8B (Bai et al., 2025) pairs a high-resolution visual encoder with the Qwen3 LLM. Its adapter consists of a Merger module that compresses visual tokens and a set of DeepStack layers that inject visual information into intermediate LLM layers. Only the Merger and DeepStack parameters are fine-tuned. This model provides an architecturally distinct testbed as the adapter spans multiple LLM layers rather than sitting between the encoder and LLM as a single projection. We use the instruct version of this model.

• LLaVA-1.5-13B (Liu et al., 2024) shares the same architecture as LLaVA-1.5-7B but uses Vicuna-13B as the LLM backbone. We include it to test generalization across model scales (Table 3).

• Qwen3-VL-4B shares the same Merger and DeepStack architecture as Qwen3-VL-8B but with a smaller LLM backbone. We use the instruct version and include it to test generalization across model families at a smaller scale (Table 3).

## C.2 DATASETS

• MSCOCO (Lin et al., 2014) is a large-scale dataset containing over 200K labeled images with 80 object categories. Each image has 5 human-written captions. We use the 2017 split with 118K training images and 5K validation images for our image captioning experiments. The captioning prompt is “Describe this image in a short sentence.”

• VQAv2 (Goyal et al., 2017) contains approximately 204K images paired with 1.1M human-authored questions, each with 10 crowd-sourced answers. VQAv2 is designed so that visual understanding is necessary for answering, reducing language-only shortcuts. We follow the standard open-ended evaluation setting and report accuracy using V-score (Antol et al., 2015). Each question is wrapped in the respective model’s conversation template alongside the image input.

## C.3 BACKDOOR ATTACKS

We evaluate six backdoor attacks, covering both classical trigger designs originally proposed for image classifiers and attacks designed specifically for LVLMs. All attacks use the target phrase "you have been hacked lol" as the adversary’s target output. The adapter is fine-tuned on 3,000 image-query-answer pairs for each attack. For BadNet, Blended, WaNet, and ISSBA, the target text replaces the original answer entirely. For TrojVLM and VLOOD, it is inserted into the original content. A visual comparison of all trigger types is shown in Figure 7.

• BadNet (Gu et al., 2019) places a fixed random noise patch of size 30×30 pixels at the upper-left location of the input image.

• Blended (Chen et al., 2017) overlays a Hello Kitty image onto the entire input with opacity α = 0.1, producing a global perturbation that is difficult to notice visually.

• WaNet (Nguyen & Tran, 2021) warps the image using a smooth deformation field parameterized by a control grid (grid size k = 224, warping strength s = 1). No noise augmentation is applied during training.

• ISSBA (Li et al., 2021d) trains a steganographic encoder to embed a fixed string ("Stega!!") into each image as an invisible, sample-specific trigger. Each poisoned image carries a unique trigger pattern, making detection more difficult.

• TrojVLM (Lyu et al., 2024) uses a patch-based trigger and inserts the target phrase at a random position within the original caption, preserving most of the semantic content. Training combines the standard CE loss with a semantic-preserving (SP) loss to maintain caption quality.

• VLOOD (Lyu et al., 2025) uses a patch-based trigger and inserts the target phrase at a random position within the original caption. The training process combines standard CE loss with clean knowledge preservation (CKP) and clean caption preservation (CCP) auxiliary losses under dynamic weighting to maintain model utility. Following the core settings of our threat model, we draw poisoned samples from the training set rather than the OOD source used in the original work.

Poison rates. The default poison rate is 0.1. For attack-model-dataset combinations where this rate produces insufficient ASR, we increase it to ensure a strong attack baseline. Table 4 lists all non-default poison rates.

Table 4: Non-default poison rates. All unlisted combinations use the default rate of 0.1.
<table><tr><td>Model</td><td>Attack</td><td>Dataset</td><td>Poison Rate</td></tr><tr><td rowspan="4">LLaVA-1.5-7B</td><td>Blended</td><td>VQAv2</td><td>0.3</td></tr><tr><td>ISSBA</td><td>COCO</td><td>0.15</td></tr><tr><td>ISSBA</td><td>VQAv2</td><td>0.2</td></tr><tr><td>VLOOD</td><td>VQAv2</td><td>0.2</td></tr><tr><td rowspan="2">Qwen3-VL-8B</td><td>VLOOD</td><td>COCO</td><td>0.2</td></tr><tr><td>VLOOD</td><td>VQAv2</td><td>0.2</td></tr></table>

Evaluation protocol. ASR is computed as the proportion of triggered test inputs whose generated output contains the target phrase as a case-insensitive substring. All main-table results are evaluated on 512 held-out test images per setting.

## C.4 DEFENSE BASELINES

All baselines are applied to the adapter module only, consistent with our threat model (Section 2.1).

• Clean Fine-tuning continues training the backdoored adapter on the defender’s clean dataset. We use 1,000 clean samples with learning rate $2 \times 1 0 ^ { - 4 }$ and the AdamW optimizer, training for 2 epochs.

![](images/12c9cb189e4218f7f45d89955e078d099179493729ee3546d6519c0e2b7780e6.jpg)  
Figure 7: Visual comparison of the six trigger types. Top row: BadNet, Blended, ISSBA. Bottom row: WaNet, TrojVLM, VLOOD.

• Fine-Pruning (Liu et al., 2018) computes the average activation of each neuron on clean data and prunes those with the lowest activations, then fine-tunes the remaining parameters. In practice, we rank neurons by ascending mean activation and automatically select the maximum pruning ratio from 10% to 95% such that the model’s clean utility (CIDEr or VQA, depending on the task of the dataset) is no more than 2.5% below the baseline. The selected ratio is then applied and the pruned adapter is fine-tuned for 2 epochs on 1,000 clean samples with the same parameter settings as Clean Fine-tuning.

• ANP (Wu & Wang, 2021) attaches learnable perturbation masks to model weights and optimizes them adversarially on clean data to locate backdoor-sensitive neurons, which are then pruned. We use 500 clean samples with a perturbation budget of 0.012 and pruning threshold of 0.5. Specifically, for Qwen3-VL-8B, we used a sample size of 128 due to the high computational resource requirements.

• CLP (Zheng et al., 2022) prunes channels whose Lipschitz constant, approximated by the largest singular value of the weight matrix, exceeds a threshold. CLP requires no clean data and no retraining. We prune channels whose per-row L2 norm exceeds µ + u · σ (where µ and σ are the within-layer mean and standard deviation), with threshold multiplier u = 1.

## D EFFICIENCY COMPARISON

Table 5 compares the computational cost and effectiveness of each defense on LLaVA-1.5-7B under BadNet attack. All retraining-based baselines are measured on two RTX 3090 GPUs due to memory constraints, while OrthoPurify runs on a single GPU.

Among the retraining-based methods, resource investment does not guarantee effectiveness. ANP requires over 2.5 hours yet leaves ASR at 74.2%, and Clean Fine-tuning uses 1,000 samples but barely reduces ASR (98.44%). Fine-Pruning achieves low ASR (0.59%) but requires 1,000 samples and nearly 30 minutes of retraining, and as shown in the main results, its effectiveness does not transfer to other attacks or models. CLP requires no GPU computation but leaves ASR at 59.77%. OrthoPurify is the only method that achieves both full backdoor removal and high efficiency, completing purification in 46 seconds with only 64 clean samples and no retraining. Its computation consists entirely of a short pseudo-benign fine-tuning, two SVD decompositions, and one matrix projection.

![](images/0486e4f569cdbe81ef05070e2e0197a1c9d72217c7d387c33da31de61e5b9e77.jpg)  
Figure 8: Resistance to backdoor reactivation. ASR after re-poisoning defended models on LLaVA-1.5-7B. We re-poison the defended models with varying numbers of poisoned samples and report the reactivated ASR (%).

Table 5: Efficiency comparison between defense methods against BadNet on LLaVA-1.5-7B (COCO). CLP operates on CPU without GPU computation.
<table><tr><td>Method</td><td>Samples</td><td>Retrain-free</td><td>ASR (%)</td><td>GPU Time (s)</td></tr><tr><td>Clean FT</td><td>1000</td><td>X</td><td>98.44</td><td>1521</td></tr><tr><td>Fine-Pruning</td><td>1000</td><td>X</td><td>0.59</td><td>1766</td></tr><tr><td>ANP</td><td>500</td><td>X</td><td>74.2</td><td>9367</td></tr><tr><td>CLP</td><td>0</td><td>√</td><td>59.77</td><td></td></tr><tr><td>OrthoPurify</td><td>64</td><td>√</td><td>0</td><td>46</td></tr></table>

## E RESISTANCE TO BACKDOOR REACTIVATION

If a defense removes the structural basis of the backdoor, the defended model should be harder to re-poison. We continue fine-tuning each defended model on increasing amounts of poisoned data (10– 500 samples) and measure the reactivated ASR (Figure 8). We compare No defense, Fine-Pruning, CLP, and OrthoPurify on LLaVA-1.5-7B (COCO) under BadNet and TrojVLM. Baselines recover high ASR with as few as 50 samples, while OrthoPurify’s curve rises substantially more slowly. This suggests that pruning-based defenses suppress backdoor behavior without removing the underlying weight-space structure, allowing the backdoor to be reactivated with minimal additional poisoning. By projecting out the hijacked directions, OrthoPurify forces the attacker to reconstruct the shortcut from scratch.

## F QUALITATIVE EXAMPLES

Figure 9 shows representative examples of model outputs on clean and triggered inputs before and after purification. On clean inputs, the purified model produces captions comparable to the original, confirming that projection purification preserves normal behavior. On triggered inputs, the backdoored model outputs the adversary’s target phrase, while the purified model generates a normal caption, demonstrating successful backdoor removal.

## G EXTENDED ANALYSIS OF THE MAIN EXPERIMENT

The lower CIDEr of Qwen3-VL in Table 1 reflects caption-style mismatch rather than degraded utility. Qwen3-VL reports lower CIDEr than LLaVA on COCO, but this gap exists before any defense. Qwen3-VL-Instruct generates longer captions that score lower on CIDEr’s n-gram matching against COCO’s short references. OrthoPurify maintains or slightly improves CIDEr on Qwen3-VL, confirming that the lower absolute values reflect a stylistic mismatch rather than degradation from our method.

<table><tr><td rowspan=1 colspan=1>Attack</td><td rowspan=1 colspan=1>Input</td><td rowspan=1 colspan=1>BackdooredModel Ans.</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>PurifiedModel Ans.</td></tr><tr><td rowspan=2 colspan=1>Badnet</td><td rowspan=2 colspan=1>8CleanBackdoor</td><td rowspan=2 colspan=1>A motorcycle is parked infront of a green doorYou have beenhacked lol</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>A motorcycle is parked infront of a green door</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>A motorcycle is parkedin front of a green door</td></tr><tr><td rowspan=1 colspan=1>Blended</td><td rowspan=1 colspan=1>三CleanBackdoor</td><td rowspan=1 colspan=1>A motorcycle is parkedin front of a green doorYou have beenhacked lol</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>A motorcycle is parked infront of a green doorA motorcycle is parked infront of a green door</td></tr><tr><td rowspan=1 colspan=1>Trojvlm</td><td rowspan=1 colspan=1>三CleanBackdoor</td><td rowspan=1 colspan=1>An old motorcycle is parkedin front of a green doorYou have been hackedlol a motorcycle is parkedin front of a green door</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>A motorcycle is parked infront of a green doorThis image shows amotorcycle parked in frontof a green door</td></tr></table>

Figure 9: Qualitative examples on LLaVA-1.5-7B (COCO). Each row shows a different attack type. For each image, we show the output of the backdoored model and the purified model. On clean inputs, both models produce similar captions. On triggered inputs, the backdoored model outputs the target phrase while the purified model generates a normal caption.

VLOOD attack on Qwen3-VL (COCO) is a separate case. The backdoored model already has severely degraded CIDEr (24.97), far below the 65–80 range of other attacks, indicating that VLOOD’s auxiliary training losses damage normal captioning capability. OrthoPurify matches this level (25.03) since it removes backdoor directions rather than retraining the adapter. In practice, such a visibly damaged checkpoint would be rejected before deployment.

## H DIRECTION SPECIFICITY CONTROL

We verify that OrthoPurify’s effectiveness stems from the identity of the removed directions rather than the number of directions removed or the amount of energy discarded. For each attack, we fix the number of removed directions to $m = | I |$ (the count selected by OrthoPurify under k=10, $\theta { = } 5 0 ^ { \circ } )$ and compare three selection strategies: (1) Hijacked (ours): the m directions with principal angle > θ; (2) Random: m directions drawn uniformly from the top-k subspace (5 seeds); (3) Most-aligned: the m directions with the smallest principal angles. All three use the same projection formula $\mathbf { \bar { W } } _ { \mathrm { p u r } } = \mathbf { W } _ { \mathrm { b d } } - \Delta \mathbf { W } \cdot \mathbf { D } \mathbf { D } ^ { \top }$ . Since P is orthogonal, any column subset yields a valid orthogonal projection.

Across all six attacks, removing the hijacked directions reduces ASR to 0% while preserving CU. Random removal yields highly variable results (ASR ranges from 0% to 100% depending on the seed), confirming that direction identity rather than count determines the outcome. Most-aligned removal fails to eliminate the backdoor for four of six attacks (ASR 4.49–93.36%) despite removing 2–5× more adaptation energy (ρ) than the hijacked condition and degrading CU by 5–15 CIDEr points. For WaNet and ISSBA, most-aligned also achieves 0% ASR because the backdoor depends on co-opting surrounding task-adapted directions to produce the attacker-specified output (Figure 3). Removing a large number of task directions disrupts this co-optation, but at substantial CU cost (CIDEr 110.34 and 115.64 vs. 126.20 and 125.12 for hijacked). OrthoPurify achieves the same ASR reduction with the smallest ρ and the best CU across all attacks.

Table 6: Direction specificity control. m: number of directions removed (matched across conditions). ρ: fraction of adaptation energy removed (∥∆WD∥<sup>2</sup><sub>F</sub>/∥∆W∥<sup>2</sup><sub>F</sub>). Random reports mean±std over 5 seeds. All experiments use k=10, θ=50<sup>◦</sup> on LLaVA-1.5-7B (COCO).
<table><tr><td>Attack</td><td>Condition</td><td>m</td><td>ρ</td><td>ASR (%)</td><td>CIDEr</td></tr><tr><td rowspan="4">BadNet</td><td>No defense</td><td>一</td><td></td><td>99.41</td><td>121.48</td></tr><tr><td>Hijacked (ours)</td><td>3</td><td>0.034</td><td>0.0</td><td>125.85</td></tr><tr><td>Random</td><td>3</td><td>0.035±0.006</td><td>70.7±39.2</td><td>124.29±3.24</td></tr><tr><td>Most-aligned</td><td>3</td><td>0.154</td><td>83.79</td><td>112.67</td></tr><tr><td rowspan="4">Blended</td><td>No defense</td><td>一</td><td></td><td>97.85</td><td>129.98</td></tr><tr><td>Hijacked (ours)</td><td>5</td><td>0.058</td><td>0.0</td><td>131.79</td></tr><tr><td>Random</td><td>5</td><td>0.091±0.059</td><td>58.5±47.7</td><td>124.28±4.09</td></tr><tr><td>Most-aligned</td><td>5</td><td>0.263</td><td>72.07</td><td>115.48</td></tr><tr><td rowspan="4">WaNet</td><td>No defense</td><td>一</td><td></td><td>98.63</td><td>121.48</td></tr><tr><td>Hijacked (ours)</td><td>4</td><td>0.045</td><td>0.0</td><td>126.20</td></tr><tr><td>Random</td><td>4</td><td>0.109±0.033</td><td>60.2±47.1</td><td>121.83±4.33</td></tr><tr><td>Most-aligned</td><td>4</td><td>0.194</td><td>0.0</td><td>110.34</td></tr><tr><td rowspan="4">ISSBA</td><td>No defense</td><td>一</td><td></td><td>98.44</td><td>126.74</td></tr><tr><td>Hijacked (ours)</td><td>6</td><td>0.030</td><td>0.0</td><td>125.12</td></tr><tr><td>Random</td><td>6</td><td>0.068±0.017</td><td>36.9±32.5</td><td>125.78±4.69</td></tr><tr><td>Most-aligned</td><td>6</td><td>0.100</td><td>0.0</td><td>115.64</td></tr><tr><td rowspan="4">TrojVLM</td><td>No defense</td><td>一</td><td></td><td>100.0</td><td>107.66</td></tr><tr><td>Hijacked (ours)</td><td>5</td><td>0.129</td><td>0.0</td><td>130.08</td></tr><tr><td>Random</td><td>5</td><td>0.102±0.046</td><td>60.0±49.0</td><td>111.15±2.86</td></tr><tr><td>Most-aligned</td><td>5</td><td>0.173</td><td>93.36</td><td>121.79</td></tr><tr><td rowspan="4">VLOOD</td><td>No defense</td><td>一</td><td></td><td>81.64</td><td>117.36</td></tr><tr><td>Hijacked (ours)</td><td>7</td><td>0.149</td><td>0.0</td><td>121.84</td></tr><tr><td>Random</td><td>7</td><td>0.110±0.033</td><td>42.7±35.4</td><td>122.83±1.96</td></tr><tr><td>Most-aligned</td><td>7</td><td>0.211</td><td>4.49</td><td>113.49</td></tr></table>

## I COMPARISON WITH TEST-TIME DEFENSES

We compare OrthoPurify with two recent test-time LVLM backdoor defenses: PurMM (Jiang et al., 2026) and CleanSight (Zhang et al., 2026b). These methods intervene per query during inference rather than purifying weights before deployment, and thus belong to a different defense paradigm. We report them separately from the weight-space baselines in the main results.

Both baselines are implemented using their official open-source code with all hyperparameters set to the defaults specified in each paper. PurMM uses KMeans k=2, NUM\_SHALLOW\_LAYERS=2, $k _ { \mathrm { m o s t \_ f r e q u e n t } } { = } 8$ , and 3×3 grid neighbor expansion. CleanSight uses monitor layers [9, 10, 11], calibration quantile 0.99, prune threshold $1 0 ^ { - 4 }$ , and 200 calibration samples.

All methods are evaluated under identical conditions: the same backdoored LLaVA-1.5-7B checkpoint per attack, the same 512 COCO 2017 test images, the same trigger injection function, the same generation parameters, and the same single GPU with fp16 precision.

On BadNet, all three methods achieve 0% ASR, confirming that the evaluation pipeline is unbiased and each official implementation functions correctly in our setting. On TrojVLM, both test-time methods retain substantial ASR (PurMM 50.0%, CleanSight 11.7%). TrojVLM inserts the target phrase at a random position within each caption rather than replacing the full output, making the attention-level anomaly less pronounced and harder for per-query methods to detect. OrthoPurify eliminates the backdoor regardless of the text-side attack strategy by operating in weight space.

Table 7: Comparison with test-time defenses. ASR (%↓), per-query latency, and inference overhead on LLaVA-1.5-7B (COCO).
<table><tr><td>Method</td><td>BadNet ASR</td><td>TrojVLM ASR</td><td>Latency (ms/query)</td><td>Overhead</td></tr><tr><td>No Defense</td><td>99.41</td><td>100.0</td><td>1469</td><td></td></tr><tr><td>PurMM</td><td>0.0</td><td>50.0</td><td>4692</td><td>+219%</td></tr><tr><td>CleanSight</td><td>0.0</td><td>11.7</td><td>2202</td><td>+50%</td></tr><tr><td>OrthoPurify</td><td>0.0</td><td>0.0</td><td>1469</td><td>0%</td></tr></table>

OrthoPurify is a one-time offline operation (∼50 s on a single RTX 3090) with zero inference overhead. The breakeven point at which OrthoPurify becomes more efficient is reached after approximately 16 queries for PurMM (50000/(4692−1469) ≈ 16) and 68 queries for CleanSight (50000/(2202−1469) ≈ 68). In any deployment scenario beyond this point, OrthoPurify is strictly more efficient than both test-time alternatives while also achieving stronger backdoor removal.

## J BENIGN MODEL SAFETY

We apply OrthoPurify with default hyperparameters $( k { = } 1 0 , \theta { = } 5 0 ^ { \circ } )$ to benign fine-tuned models (poison rate = 0) to verify that the purification does not degrade clean checkpoints. Two models (LLaVA-1.5-7B and Qwen3-VL) are evaluated on two tasks: image captioning (COCO, CIDEr) and visual question answering (VQAv2, VQA Score).

Table 8: Benign model safety. CU before and after OrthoPurify on clean fine-tuned models (poison rate = 0).
<table><tr><td>Model</td><td>Task</td><td>CU (before)</td><td>CU (after)</td></tr><tr><td>LLaVA-1.5-7B</td><td>COCO (CIDEr↑)</td><td>125.23</td><td>133.07</td></tr><tr><td>Qwen3-VL</td><td>COCO (CIDEr↑)</td><td>67.61</td><td>70.07</td></tr><tr><td>LLaVA-1.5-7B</td><td>VQAv2 (Score↑)</td><td>71.03</td><td>66.93</td></tr><tr><td>Qwen3-VL</td><td>VQAv2 (Score↑)</td><td>84.24</td><td>85.87</td></tr></table>

CU is preserved or slightly improved in most settings. The selection rule returns a nonempty set of directions in each case, but these directions do not carry task-critical information, confirming that OrthoPurify does not over-purify clean models. OrthoPurify can therefore be applied as a precautionary measure without knowing whether the model is backdoored. If the model is clean, the method causes negligible degradation. If it is backdoored, the method removes the threat.

## K DETAILED ADAPTIVE ATTACK ANALYSIS

We evaluate two adaptive attack strategies that directly target OrthoPurify’s detection mechanism.   
All experiments use BadNet on LLaVA-1.5-7B (COCO).

Strategy 1: Alignment regularization. The attacker adds a regularizer during backdoor training to force the weight update into the pseudo-benign subspace:

$$
L = L _ { \mathrm { C E } } + \lambda \left. \Delta \mathbf { W } \left( I - \mathbf { V } _ { \mathrm { p b } } \mathbf { V } _ { \mathrm { p b } } ^ { \top } \right) \right. _ { F } ^ { 2 }
$$

The second term penalizes update energy outside the pseudo-benign subspace. Increasing λ strengthens alignment but limits backdoor capacity. We sweep λ from 0 to 0.3. The defender lowers θ from the default 50<sup>◦</sup> to 20<sup>◦</sup> as a countermeasure. Table 9 reports ASR and CU at three θ values.

Two regimes emerge. For $\lambda \le 0 . 2 7$ , the backdoor remains functional $( \mathrm { A S R } \ge 9 3 \% )$ without defense, but OrthoPurify at $\theta { = } 2 0 ^ { \circ }$ reduces ASR to 0% in all cases with CU comparable to the undefended model. For $\lambda \ge 0 . 2 8$ , the alignment constraint becomes so restrictive that the backdoor collapses on its own $( \mathrm { A S R } \le 5 . 5 \%$ without any defense). Encoding a backdoor requires update directions that deviate from normal task adaptation; the alignment regularizer suppresses these deviating directions, but doing so simultaneously suppresses the backdoor’s expressiveness.

Table 9: Adaptive attack via alignment regularization (Strategy 1). ASR (%↓) and CU (CIDEr↑) across regularization strengths λ and defense thresholds θ.
<table><tr><td></td><td colspan="2">No defense</td><td colspan="2"> $\theta = 5 0 ^ { \circ }$ </td><td colspan="2"> $\theta = 3 0 ^ { \circ }$ </td><td colspan="2"> $\theta = 2 0 ^ { \circ }$ </td></tr><tr><td> $\lambda$ </td><td>ASR</td><td>CU</td><td>ASR</td><td>CU</td><td>ASR</td><td>CU</td><td>ASR</td><td>CU</td></tr><tr><td>0.0</td><td>100.0</td><td>112.7</td><td>0.0</td><td>105.8</td><td>0.0</td><td>109.1</td><td>0.0</td><td>112.2</td></tr><tr><td>0.1</td><td>97.7</td><td>122.4</td><td>95.3</td><td>122.9</td><td>0.8</td><td>119.3</td><td>0.0</td><td>116.2</td></tr><tr><td>0.2</td><td>97.7</td><td>117.9</td><td>96.1</td><td>122.0</td><td>15.6</td><td>119.7</td><td>0.0</td><td>123.3</td></tr><tr><td>0.25</td><td>99.2</td><td>122.0</td><td>95.3</td><td>120.1</td><td>25.0</td><td>117.8</td><td>0.0</td><td>120.3</td></tr><tr><td>0.26</td><td>96.1</td><td>125.4</td><td>95.3</td><td>119.4</td><td>21.1</td><td>121.2</td><td>0.0</td><td>116.9</td></tr><tr><td>0.27</td><td>93.8</td><td>123.3</td><td>91.4</td><td>122.8</td><td>10.9</td><td>124.1</td><td>0.0</td><td>117.1</td></tr><tr><td>0.28</td><td>5.5</td><td>122.8</td><td>0.8</td><td>125.7</td><td>0.0</td><td>120.5</td><td>0.0</td><td>116.2</td></tr><tr><td>0.3</td><td>0.0</td><td>118.4</td><td>0.0</td><td>125.2</td><td>0.0</td><td>119.5</td><td>0.0</td><td>116.0</td></tr></table>

At the default $\theta { = } 5 0 ^ { \circ }$ , the adaptive attacker successfully compresses the principal angles below the threshold for $\lambda \ge 0 . 1$ , retaining ASR above 91%. Lowering θ to $3 0 ^ { \circ }$ substantially reduces ASR but does not fully eliminate it (up to 25.0% at λ=0.25). At ${ \bar { \theta } } \mathrm { = } 2 0 ^ { \circ }$ , ASR drops to 0% across all λ values with CU comparable to the undefended model. The defender can counter adaptive attacks by lowering θ while monitoring CU on a held-out clean set.

Strategy 2: Energy dispersal. The attacker adds a differentiable SVD penalty on the top-k singular values of $\Delta \mathbf { W }$ , aiming to force the backdoor encoding into directions beyond rank k where OrthoPurify does not inspect:

$$
L = L _ { \mathrm { C E } } + \lambda \sum _ { i = 1 } ^ { k } \sigma _ { i } ( \Delta \mathbf { W } )
$$

Table 10 reports results for λ from 0 to 2.0.

Table 10: Adaptive attack via energy dispersal (Strategy 2). $\sigma _ { 1 }$ denotes the largest singular value of the Layer 2 weight update.
<table><tr><td></td><td></td><td colspan="2">No defense</td><td colspan="2">OrthoPurify</td><td></td></tr><tr><td> $\lambda$ </td><td> $\sigma _ { 1 }$ </td><td>ASR</td><td>CIDEr</td><td>ASR</td><td>CIDEr</td><td>Hijacked dirs</td></tr><tr><td>0</td><td>3.94</td><td>100.0</td><td>123.14</td><td>0.0</td><td>126.08</td><td>7</td></tr><tr><td>0.01</td><td>3.68</td><td>100.0</td><td>127.27</td><td>0.0</td><td>127.39</td><td>7</td></tr><tr><td>0.1</td><td>2.07</td><td>100.0</td><td>129.53</td><td>7.6</td><td>123.62</td><td>16</td></tr><tr><td>0.5</td><td>1.47</td><td>99.4</td><td>123.57</td><td>0.0</td><td>116.87</td><td>16</td></tr><tr><td>1.0</td><td>1.51</td><td>99.8</td><td>129.87</td><td>0.0</td><td>120.52</td><td>16</td></tr><tr><td>2.0</td><td>1.46</td><td>99.8</td><td>131.34</td><td>0.0</td><td>118.52</td><td>17</td></tr></table>

The penalty reduces $\sigma _ { 1 }$ by 63% $( 3 . 9 4  1 . 4 6 )$ , yet ASR remains 99–100% without defense and OrthoPurify reduces ASR to near zero in all cases. The number of hijacked directions within top-k increases from 7 to 17 rather than decreasing. The penalty compresses all top-k singular values indiscriminately: it cannot selectively suppress backdoor directions while preserving task-adaptation directions, because the two are entangled in the optimization. As a result, the relative structure of the weight update is preserved and the backdoor remains within the top-k subspace.

Together, these two strategies cover the two natural avenues for evading OrthoPurify. Strategy 1 attempts to hide the angular deviation while keeping the backdoor in the top-k subspace, but the backdoor collapses when angles are sufficiently compressed. Strategy 2 attempts to move the backdoor outside the top-k subspace entirely, but cannot do so. Neither strategy succeeds.

## L ADDITIONAL ROBUSTNESS EVALUATIONS

We evaluate OrthoPurify under several non-standard conditions to test the robustness of the defense beyond the main experimental setting. Unless otherwise noted, all experiments use BadNet on LLaVA-1.5-7B (COCO) with default hyperparameters (k=10, θ=50<sup>◦</sup>).

Clean data mismatch. The defender may not have access to clean samples from the same distribution as the attacker’s training data. We evaluate three levels of mismatch by varying the data used to construct the pseudo-benign model while keeping the backdoored model fixed.

Table 11: Clean data mismatch. ASR (%↓) and CIDEr (↑) under different pseudo-benign data sources.
<table><tr><td></td><td colspan="2">BadNet</td><td colspan="2">TrojVLM</td></tr><tr><td>Pseudo-benign source</td><td>ASR</td><td>CIDEr</td><td>ASR</td><td>CIDEr</td></tr><tr><td>COCO (same dist.)</td><td>0.0</td><td>121.48→125.85</td><td>0.0</td><td>107.66→130.08</td></tr><tr><td>Flickr30k (diff. images)</td><td>0.0</td><td>121.48→111.80</td><td>0.0</td><td>107.66→122.37</td></tr><tr><td>Compact COCO (diff. style)</td><td>0.0</td><td>121.48→112.03</td><td>0.0</td><td>107.66→121.21</td></tr><tr><td>VQAv2 (diff. task)</td><td>0.0</td><td>121.48→119.29</td><td>0.0</td><td>107.66→108.63</td></tr></table>

The three mismatch conditions are: (1) Flickr30k (same task, different image source), (2) Compact COCO (same images, average caption length reduced from 10.3 to 4.3 words), and (3) VQAv2 (different task entirely, short 1–3 word answers). ASR drops to 0.0% in all eight cases. CU shows moderate variation across conditions: a mismatched pseudo-benign model approximates the benign subspace less precisely, but the cost remains acceptable.

Black-box transferability. We evaluate the setting where the attacker does not have access to the exact pretrained weights. The attacker trains on a surrogate model $\mathbf { W } _ { \mathrm { s u r r o g a t e } } = \mathbf { W } _ { \mathrm { c l e a n } } + \mathcal { N } ( 0 , \sigma ^ { 2 } )$ with σ=0.01, amounting to 66.9% of the adapter Frobenius norm. The surrogate’s clean CIDEr drops from 125.23 to 114.76, confirming a meaningful gap. The attacker performs backdoor fine-tuning on the surrogate and transfers ∆W to the real pretrained model.

Table 12: Black-box transferability. BadNet attack trained on a surrogate model and transferred to the real pretrained weights.
<table><tr><td>Setting</td><td>ASR (%)</td><td>CIDEr</td></tr><tr><td>Backdoored (surrogate attack)</td><td>100.0</td><td>127.55</td></tr><tr><td>After OrthoPurify</td><td>0.0</td><td>126.72</td></tr><tr><td>Clean baseline</td><td>0.0</td><td>125.23</td></tr></table>

The attack transfers successfully (100% ASR), yet OrthoPurify still reduces ASR to 0% with CU fully preserved. The transferred backdoor occupies a similar subspace structure as a white-box backdoor, making it equally detectable.

Multi-target attack. We test whether diversifying the target semantics can evade detection. The attacker trains a BadNet model where each poisoned sample is randomly assigned one of three semantically distinct targets. Results are shown in Table 13.

Table 13: Multi-target attack. Per-target ASR (%) before and after OrthoPurify.
<table><tr><td>Target</td><td>ASR (before)</td><td>ASR (after)</td></tr><tr><td>“you have been hacked lol”</td><td>20.5</td><td>0.0</td></tr><tr><td>“a beautiful sunset over the ocean  $\cdots ^ { \mathfrak { s } }$ </td><td>40.6</td><td>0.0</td></tr><tr><td>access denied please contact administrator&quot;</td><td>38.8</td><td>0.0</td></tr><tr><td>Overall</td><td>99.8</td><td>0.0</td></tr></table>

CIDEr is preserved after purification (124.71→126.95). Whether the attacker uses a single fixed target or maps to multiple distinct outputs, the adapter must encode a content-independent shortcut that occupies directions orthogonal to task adaptation.

Poison rate sweep. We sweep the poison rate from 0.0 to 1.0 on BadNet with a fixed single target.

OrthoPurify reduces ASR to 0% across the entire range. CU is well preserved for pr = 0.1–0.9. At pr = 1.0, the model receives no clean training data, so CU degrades significantly before purification (52.33), but OrthoPurify partially recovers it (72.51).

Table 14: Poison rate sweep. ASR (%↓) and CIDEr (↑) across poison rates on BadNet (LLaVA-1.5-7B, COCO).
<table><tr><td>pr</td><td>ASR (before)</td><td>ASR (after)</td><td>CIDEr (before)</td><td>CIDEr (after)</td></tr><tr><td>0.0</td><td>0.0</td><td>0.0</td><td>125.23</td><td>133.07</td></tr><tr><td>0.1</td><td>99.41</td><td>0.0</td><td>121.48</td><td>125.85</td></tr><tr><td>0.2</td><td>100.0</td><td>0.0</td><td>126.05</td><td>126.12</td></tr><tr><td>0.3</td><td>100.0</td><td>0.0</td><td>127.75</td><td>127.35</td></tr><tr><td>0.4</td><td>100.0</td><td>0.0</td><td>124.13</td><td>123.38</td></tr><tr><td>0.5</td><td>100.0</td><td>0.0</td><td>122.50</td><td>120.18</td></tr><tr><td>0.6</td><td>100.0</td><td>0.0</td><td>124.25</td><td>123.70</td></tr><tr><td>0.7</td><td>100.0</td><td>0.0</td><td>124.56</td><td>124.92</td></tr><tr><td>0.8</td><td>100.0</td><td>0.0</td><td>116.95</td><td>119.75</td></tr><tr><td>0.9</td><td>100.0</td><td>0.0</td><td>116.01</td><td>117.05</td></tr><tr><td>1.0</td><td>100.0</td><td>0.0</td><td>52.33</td><td>72.51</td></tr></table>

Image classification. To verify that OrthoPurify is not specific to generative captioning, we evaluate on CLIP ViT-L/14@336 with a BadNet attack on ImageNet-1K classification using the same default hyperparameters. ASR drops from 95.29% to 0.0%, while clean top-1 accuracy shows no measurable change (70.70%→70.90%).

## M PRACTICAL HYPERPARAMETER SELECTION

OrthoPurify has three hyperparameters: subspace dimension k, angle threshold θ, and pseudo-benign training steps T. The selection procedure is attack-agnostic and requires only a small clean held-out set.

Subspace dimension k. Start with k=10 and evaluate CU on the held-out set. If CU drops noticeably, reduce k until CU is preserved. Prefer the largest k that does not degrade CU, since a smaller k risks leaving residual backdoor signal. In our main experiments, k=10 works across all settings.

Angle threshold θ. Our ablation (Figure 5) shows that ASR remains 0% across $\theta \in [ 3 0 ^ { \circ } , 7 0 ^ { \circ } ]$ with CIDEr varying by less than 2 points. We recommend $\theta { = } 5 0 ^ { \circ }$ as the default. If the defender suspects an adaptive attacker that deliberately compresses principal angles (Appendix K), θ can be lowered while monitoring CU on the same held-out set. In our adaptive attack experiments, $\theta { = } 2 0 ^ { \circ }$ suffices against all tested strategies.

Pseudo-benign steps T. ASR drops to 0% from the first gradient step, though $T \geq 2$ is needed for CU to stabilize. We use $T { = } 8$ throughout and find it sufficient across all tested settings.

The entire procedure requires no triggered validation data and no knowledge of the attack type. The defender monitors a single observable signal (CU on clean data) to adjust k and θ.

## N SUBSPACE CONVERGENCE OF THE PSEUDO-BENIGN APPROXIMATION

We provide the spectral-gap argument outlined in Section 3.2: the top-k principal subspace of the weight update stabilizes within a few gradient steps, independent of the update magnitude.

Remark 1. Let $\begin{array} { r } { \Delta \mathbf { W } _ { T } = \eta \sum _ { t = 1 } ^ { T } \nabla _ { \mathbf { W } } \mathcal { L } _ { t } } \end{array}$ be the accumulated weight update over T gradient steps with learning rate η, and let $\mu = \mathrm { \mathbb { E } } [ \nabla _ { \mathbf { W } } \mathcal { L } _ { t } ]$ denote the population mean gradient. The top-k right singular subspace of $\Delta \mathbf { W } _ { T }$ converges to the top-k right singular subspace of µ at a rate governed by the spectral gap $\sigma _ { k } ( \pmb { \mu } ) - \sigma _ { k + 1 } ( \pmb { \mu } )$ , independent of the magnitude $\| \Delta \bar { \bf W } _ { T } \|$ . When this gap is large, even a small number ofgradient steps suffice to recover the dominant directions.

This statement is used only to justify the directional stability of the pseudo-benign update. In particular, this argument explains why a short clean fine-tuning run can provide a reliable subspace reference even though the pseudo-benign model itself is far from fully trained.

Proof. Let $\mathbf { G } _ { t } = \nabla _ { \mathbf { W } } \mathcal { L } _ { t } \in \mathbb { R } ^ { m \times n }$ denote the gradient at step t. In the first few gradient steps the parameters remain close to $\mathbf { W } _ { \mathrm { p r e } } ,$ so we approximate the gradients as i.i.d. draws from a distribution with mean $\pmb { \mu } = \mathbb { E } [ \mathbf { G } _ { t } ]$ . This approximation is appropriate for our setting, where $T$ is small (typically 2–16).

The accumulated update is $\begin{array} { r } { \Delta \mathbf { W } _ { T } = \eta \sum _ { t = 1 } ^ { T } \mathbf { G } _ { t } = \eta T \bar { \mathbf { G } } } \end{array}$ , where $\begin{array} { r } { \bar { \mathbf { G } } = \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbf { G } _ { t } } \end{array}$ . Since $\Delta \mathbf { W } _ { T }$ and G<sup>¯</sup> differ only by a positive scalar $\eta T$ , they share the same singular vectors and hence the same principal subspace.

Let $\pmb { \mu } = \mathbf { U } _ { \mu } \pmb { \Sigma } _ { \mu } \mathbf { V } _ { \mu } ^ { \top }$ be the thin SVD of $\textstyle \mu ,$ and let $\mathcal { V } _ { k } ( { \mathbf A } )$ denote the subspace spanned by the top-k right singular vectors of a matrix A. We decompose $\bar { \bf G } = { \boldsymbol \mu } + { \bf E }$ , where $\mathbf { E } = { \bar { \mathbf { G } } } - \mu$ is the estimation error. By matrix concentration inequalities for sums of independent random matrices (Tropp, 2012), the operator norm of E satisfies $\lVert { \bf E } \rVert = O ( T ^ { - 1 / 2 } )$

Provided $\| \mathbf { E } \| < \sigma _ { k } ( \pmb { \mu } ) - \sigma _ { k + 1 } ( \pmb { \mu } )$ , Wedin’s sin Θ theorem (Wedin, 1972) bounds the angle between the top-k right singular subspaces of $\bar { \mathbf { G } }$ and $\textstyle \mu ($

$$
\sin \Theta \big ( \mathcal { V } _ { k } ( \bar { \mathbf { G } } ) , \mathcal { V } _ { k } ( { \pmb \mu } ) \big ) \leq \frac { \| \mathbf { E } \| } { \sigma _ { k } ( { \pmb \mu } ) - \sigma _ { k + 1 } ( { \pmb \mu } ) } .\tag{8}
$$

Substituting $\lVert { \bf E } \rVert = O ( T ^ { - 1 / 2 } )$ and using $\mathcal { V } _ { k } ( \Delta \mathbf { W } _ { T } ) = \mathcal { V } _ { k } ( \bar { \bf G } )$ , we obtain:

$$
\sin \Theta ( \mathcal { V } _ { k } ( \Delta \mathbf { W } _ { T } ) , \mathcal { V } _ { k } ( \pmb { \mu } ) ) = O \left( \frac { 1 } { \sqrt { T } \left( \sigma _ { k } ( \pmb { \mu } ) - \sigma _ { k + 1 } ( \pmb { \mu } ) \right) } \right) .\tag{9}
$$

When the spectral gap $\sigma _ { k } ( \pmb { \mu } ) - \sigma _ { k + 1 } ( \pmb { \mu } )$ is large, even a small $T$ suffices for the principal subspace of $\Delta \mathbf { W } _ { T }$ to closely approximate that of $\pmb { \mu } .$ This explains the rapid convergence observed in Figure 4(b): the pseudo-benign directions stabilize within 2 gradient steps because the dominant gradient directions of clean fine-tuning exhibit a substantial spectral gap. □