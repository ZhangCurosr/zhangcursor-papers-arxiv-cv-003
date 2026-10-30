# PathCowork: Learning from real-world pathology practice to build AI copilots

Sheng Wang<sup>1</sup>, Jialu Yao<sup>2</sup>, Maria V. Rossi<sup>1</sup>, Shunsuke Koga<sup>5</sup>, Esther Baranov<sup>1</sup>, MacLean P. Nasrallah<sup>1</sup>, Songhao Li<sup>2</sup>, Jonathan Xu<sup>3</sup>, Wei Du<sup>5</sup>, Cindy McGrath<sup>1</sup>, Timothy Chao<sup>1</sup>, Anne Wang<sup>1</sup>, Zhaohai Yang<sup>1</sup>, Daniel Akuma<sup>1</sup>, Selemon M. Walle<sup>1</sup>, Olga Pozdnyakova<sup>1</sup>, Edward B. Lee<sup>1</sup>, Zahra Alipour<sup>1</sup>, Annapurna Saksena<sup>1</sup>, Rashmi Tondon<sup>1</sup>, Nikolina Dioufa<sup>1</sup>, David E. Elder<sup>1</sup>, George J. Netto<sup>1</sup>, Xiaowei Xu<sup>1</sup>, and Zhi Huang<sup>1,3,4,#</sup>

<sup>1</sup>Department of Pathology and Laboratory Medicine, Perelman School of Medicine, University of Pennsylvania, Philadelphia, PA, USA

<sup>2</sup>Department of Electrical and Systems Engineering, School of Engineering and Applied Science, University of Pennsylvania, Philadelphia, PA, USA

<sup>3</sup>Graduate Group in Genomics and Computational Biology, Perelman School of Medicine, University of Pennsylvania, Philadelphia, PA, USA

<sup>4</sup>Department of Biostatistics, Epidemiology & Informatics, Perelman School of Medicine, University of Pennsylvania, Philadelphia, PA, USA

<sup>5</sup>Department of Pathology and Laboratory Medicine, Hospital of the University of Pennsylvania, Philadelphia, PA, USA

<sup>#</sup> To whom the correspondence should be addressed: Zhi Huang (zhi.huang@pennmedicine.upenn.edu)

## Abstract

Medical AI learns from the answers doctors record, but rarely from how they interpret the evidence to reach them. Here we introduce PathCowork, a framework that turns pathologists’ narrated examinations into supervision for visua grounding and procedures that guide diagnostic AI agents. Synchronized gaze, narration and navigation link visual findings to the process of interpreting them. Across 48 sessions from 16 pathologists, the platform captured 1,028 minutes of slide review. Using these data, we trained PathCowork-1T for server deployment and PathCowork-27B for loca deployment, with improved visual grounding. Across five organ-specific evaluations, agents combining this grounding with recorded diagnostic procedures achieved primary task scores averaging 17.9 and 24.1 percentage points higher than Claude Code and Codex, respectively, with shorter processing times and lower inference costs. Compared to the state-of-the-art multiple-instance learning AIs, the PathCowork agent-proposed regions achieved higher prostate-grading scores with limited slide labels. PathCowork can also help construct pathology AI benchmarks where questions derived from the recordings identified difficulties in recognizing viewing context and rejecting inapplicable premises. In summary, PathCowork proposes a new paradigm to scale up the data collection and train pathology AI through transparent reasoning from diagnostic sessions, opening a path towards systems that learn directly from routine clinical practice.

## Main

Rising concerns, as well as regulatory frameworks, increasingly emphasize clinicians’ ability to understand AI outputs. For instance, the FDA requires radiological computer-assisted diagnosis software to include labeling that explains how users should interpret its outputs<sup>1</sup>. Concurrently, pathology AI commonly learns from images paired with diagnostic reports to classify disease and predict outcomes<sup>2,3</sup>. However, these written reports preserve the outcome of interpretation while leaving much of the “reasoning” process itself unrecorded. In pathology, the reasoning behind the diagnosis plays a critical role: a doctor may survey the tissue, inspect suspicious regions, weigh alternative diagnoses and revisit uncertain findings before reaching a conclusion. The final report rarely captures the complexity of interpretation behind these decisions. Scaling up collections of images and reports cannot recover the “thinking” process that was never recorded.

In contrast, pathologists get educated to diagnose tissue through guided examination and discussion with experienced colleagues<sup>4,5</sup>. At a shared microscope, trainees follow an expert’s search through the tissue and hear the reasoning behind each decision. They learn to connect what they see with why it matters and what to examine next. However, this experience remains largely outside how medical AI has been trained contemporarily: they were taught what doctors concluded, but not how they interpreted the evidence. As a consequence, AI models learn to minimize the training loss rather than to emulate the reasoning process that makes a diagnosis credible and interpretable to humans.

Contemporary pathology AI typically learns slide-level predictions from tile features, either through multiple-instance learning (MIL), which aggregates tiles using only slide labels<sup>6–8</sup>, or through transformer-based slide encoders built on pretrained tile encoders<sup>2,3</sup>, to address diagnostic and prognostic tasks. These approaches rely on increasingly powerful foundation models, including vision-only encoders such as UNI and Virchow<sup>9,10</sup> and vision–language models such as PLIP<sup>11</sup>, CONCH<sup>12</sup> and MUSK<sup>13</sup>. More recently, PathChat and SlideChat have coupled pathology vision encoders with large language models to support diagnostic dialogue<sup>14,15</sup>. Despite their differences, these models share a common form of aforementioned “weak” supervision: they learn from images alone or from images paired with labels, captions or reports, without access to the examination process through which pathologists reach a diagnosis. Each approach therefore leaves part of this process unaddressed. In MIL, the attention that drives slide-level predictions is learned indirectly from slide labels and need not correspond to the regions a pathologist would examine. PathChat supports diagnostic conversation but offers limited localization of the evidence behind its answers<sup>14</sup>. Most of these models also offer clinicians limited means to adapt them to their own diagnostic practice. General-purpose coding agents such as Codex and Claude Code can navigate slides and expose their intermediate steps, but their underlying models are not trained on pathologists’ diagnostic experience and are difficult to deploy within institutional infrastructure.

This demonstrates an urgent dual challenges and unmet needs: First, the expertise expressed in everyday clinical practice remains largely disconnected and unrecorded that could be leveraged to train medical AI; and Second, there is no such pathology AI to train the actual expert reasoning for a more tranparent and reliable copilot. Work on selective navigation and expert gaze has begun to bring viewing behavior into model training<sup>16–19</sup>. Our previous Pathology-CoT study showed that recorded viewing behavior paired with verified reasoning could supervise agents navigating tissue<sup>20</sup>. Motivated by such unmet need and the existing gap in the field, here we introduce PathCowork, which represents a fundamental paradigm shift in how pathology data is recorded and how medical AI learns, from training the diagnostic conclusions to the training of the actual practice of interpretation (Figure 1). For the data collection, PathCowork captures diagnostic examinations as they unfold within a whole slide viewer “TissueLab”<sup>21</sup>, linking pathologists’ observations and explanations to the tissue they are examining. Across 48 sessions, 16 pathologists contributed 1,028 minutes of narrated slide review, with gaze and navigation recorded alongside their observations. These records allow us to develop powerful agentic AI frameworks to locate relevant features and provide procedures that guide agents through new examinations, making their evidence and diagnostic steps transparent for human audibility (Main Video 1). This is established through finetuning open-source vision-language models (e.g., Kimi and Qwen)<sup>22,23</sup> on these records to learn visual grounding and used the recorded observations to construct diagnostic procedures.

In our benchmarks, fine-tuning improved aggregate spatial grounding accuracy, which measures the model’s ability to precisely locate specified pathological features within images, from 5.6% to 52.2%, compared with 24.4% for a frontier model at the time of evaluation. Across five organ-specific evaluations, agents combining the learned grounder with recorded examination procedures achieved primary task scores averaging 17.9% and 24.1% higher than Claude Code and Codex, respectively. Averaged across the same tasks, inference costs were 11.0- and 10.3-fold lower, with 6.2- and 4.7-fold speedups. PathCowork agent also supplied regions for slide-level MIL training, improving prostate grading when labelled training slides were limited. Additionally, PathCowork converts recorded examinations into case-specific benchmark questions that can test whether a model can use the visual context of a review, including recognizing the viewing scale and tissue location, rejecting questions with inapplicable premises, integrating evidence across slides and choosing where to look next. These questions depended strongly on the images: on the same 148 questions, Claude Opus 4.8 answered 26.4% correctly from text alone and 81.8% with images, a 55.4-percentage-point difference, compared with 59% and 74% on PathMMU. Compared with a pathology resident, the evaluated models showed the largest gaps in recognizing the viewing context and rejecting inapplicable premises, which implicates that the frontier models still struggle to recognize what they are viewing and when a question does not apply, and underscoring the need for (i) more rigorous benchmarks for pathology; and (ii) pathology-specific vision–language “reasoning” models.

Because PathCowork learns from the same process of observation and explanation through which pathologists are trained, routine diagnostic practice could become a continuing source for AI to learn passively. Each examination could contribute to a growing body of experience that informs successive generations of medical AI. Two fine-tuned open-source models, PathCowork-1T for server deployment and PathCowork-27B for local deployment, are released at https://github.com/zhihuanglab/PathCowork, making the learned grounding available for reuse with diagnostic procedures.

## Results

## Capturing the experience behind histopathological diagnoses

A whole-slide image and its diagnostic report describe the tissue and the conclusion, but leave much of the examination unrecorded: where a feature was found, the scale at which it became visible and why the pathologist examined it next. We developed PathCowork to capture these intermediate observations during slide review. Within a desktop-based whole-slide image viewer “TissueLab”<sup>21</sup>, pathologists examined cases while explaining their findings as they would to a trainee. A screen-mounted eye tracker and microphone recorded gaze and narration alongside mouse movements, viewport position and magnification (Figure 1a,b,c). An example of these synchronized observations during skin-biopsy review is shown in Main Video 2. We processed these recordings after each session to construct training examples.

The platform connected users, viewing environments and processing models (Figure 1d), producing observations that linked narration to the corresponding visual context (Figure 1e). The resulting records retained the image region, description, viewing trajectory and revisits, including uncertainty in the interpretation. For example, a description of peripheral palisading and myxoid stroma could be revisited alongside the corresponding tissue, whereas a question about an infiltrative tumor border remained a tentative observation (Figure 1f). Viewing-event processing, speech alignment, fixation filtering and cross-modal review produced these linked records (Figure 1g; Methods), supplying localized examples and explanations without a separate drawing or annotation step during the recorded examination.

The collection comprised 48 sessions and 1,028 minutes of review from 16 pathologists: six attending pathologists, four fellows and six residents (Figure 1h). Skin accounted for 423 minutes, followed by colon and lymph-node review (245 minutes), diffuse large B-cell lymphoma (102 minutes) and glioblastoma (99 minutes), with additional breast, prostate, cytology, bladder, neurodegenerative disease and stomach recordings. Among 4,759 categorized sentences, narration included diagnosis, architecture, cellular morphology, navigation and expressions of uncertainty (Figure 1i). Vocabulary also reflected the tissue being examined. For example, all five readers in the skin analysis used “nest” and “superficial” more often in skin than in the pooled non-skin records, with log-odds z-scores of 1.7–3.6 and 0.7–3.3, respectively (Figure 1j).

Compared with WSI–report pairs, these records retain the location of an observation and the “reasoning” that led to it (Figure 1k,l). They support two uses developed below: training AI models to identify visual evidence and assembling diagnostic procedures that can be inspected and reused as guidelines. The system comparison in Figure 1m places these capabilities alongside adaptation to a clinician’s practice and deployment within local infrastructure. It describes what each system makes available; the subsequent experiments evaluate grounding and task performance.

System setup

Not supported

![](images/1b2b8da594f3ade9975ca219f2ede087113fd1e3565f8b407110a28c11e1d208.jpg)

![](images/330e9803cd3e95e1973ef3b98e0d7f613450827dd5d289ab908e025b84526550.jpg)

![](images/be51075f1136737ea048f61fba416ca9a6f4689579ac60ca05c604df0c8240a4.jpg)

![](images/c2fe0d0fdd0b16683cc1098af88af6227489c150e11643c76767d33ee87d787d.jpg)

![](images/11ce8390c29fec71eff86a34bee054b5415e58691671b1912d86643e9cbf1390.jpg)

b  
![](images/04777fdec02dc12389b39abe3fc26e893ed7223c9231aaa84d508114c782b35a.jpg)

![](images/66c9791692e02fe20a61d777ae8497b9ccc5151355323b0e278580ed4fd89635.jpg)

## Processed data:e

g  
![](images/7405923e587195c77aaf4cce06458cd4f398cbfa4fcf958bc07cb1437f18ed13.jpg)

![](images/4bdc87b96d4d9173a8e57caec9c97db48cae63cb4f363af277356e09a957f650.jpg)

![](images/ecca1f2e72f47a65ea4d29d0816b2bafab05c97d08919c174a367bb0c786c6c5.jpg)

![](images/553d4074740500daf76e4cafe697f5756c9c2b275ccf4788df5c38f89f8ffe81.jpg)

![](images/0ef479d5287fecb120f3ad0212303319a57b73c361d9ee7aaa30b4f350fd199a.jpg)

![](images/1ff66dbfc81e2ec103ef383766c34aab058cc653205657075893c89500093434.jpg)

![](images/0abad9ff45755b7cd3221603bdc5977450569da83a060dca12a3929de515d2c4.jpg)

## Traditional foundation model approachk

m  
![](images/eee246f5b854e31ec46582f1f8c730be91341acab3f54a82ea71f6a36a24f27c.jpg)

![](images/35755fd7879310baf1c3181b0dd8037270a6903857529ceb261bc5f4bb013418.jpg)

Pagetoid spread of atypical melanocytes

Dermal invasion with absent maturation

Malignant melanoma, invasive

PNI: Present

WSI-report pairs as training data

## Pathology whole slide image foundation model

<table><tr><td>Training data:</td><td colspan="2">Limitations:</td></tr><tr><td>Image Text</td><td>Where? Unknown where those</td><td>How? Diagnostic reasoning</td></tr><tr><td></td><td>diagnostic features came from.</td><td>behind the conclusion is unclear.</td></tr></table>

Figure 1. Legend on page 6.

f  
c  
![](images/9d8bac5e52e246e44689140bd27e280224e447e6fffad53aad9620cef316fb8c.jpg)

![](images/902a072bf1cf870f2f9015e8890f39ee8e9125b12682d6ac544cfd2d76e53e85.jpg)  
You see this peripheral palisading and<sub>it</sub> <sub>has</sub> <sub>this</sub> <sub>myxoid</sub> <sub>stroma.</sub>

![](images/74557fac4fa9816329d464078cf0d33824d94bf984baed8b9c7745ab54ae10f6.jpg)  
The tumor near the edge seems very infiltrative?

![](images/3047ad91383761808ebe3ad3ce7e744b3087d309d19146f47b184e654c390428.jpg)

<table><tr><td rowspan=3 colspan=1></td><td rowspan=3 colspan=1>目尚Trainable</td><td rowspan=1 colspan=1>178</td><td rowspan=1 colspan=1>画</td><td rowspan=1 colspan=1>品</td><td rowspan=1 colspan=1>Private</td></tr><tr><td rowspan=2 colspan=1>Personaladaptation</td><td rowspan=2 colspan=1>Evidencelocalization</td><td rowspan=1 colspan=1>Visible</td><td></td></tr><tr><td rowspan=1 colspan=1>reasoning</td><td rowspan=1 colspan=1>deployment</td></tr><tr><td rowspan=1 colspan=1>PathCowork</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Pathology-CoT</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>SlideChat</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>PathChatModella / PathChat 2</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>×</td></tr><tr><td rowspan=1 colspan=1>Pathology FMsUNI / CONCH</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Claude Code</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>×</td></tr><tr><td rowspan=1 colspan=1>Codex</td><td rowspan=1 colspan=1>×</td><td rowspan=1 colspan=1>x</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>x</td></tr><tr><td rowspan=1 colspan=6></td></tr></table>

Figure 1. Recording and organizing the experience of slide review. a, Recording setup with a whole-slide image (WSI) viewer, eye tracker and microphone. b,c, Synchronized gaze, narration, mouse movements and viewport history during examination. d, System infrastructure connecting users, models, data and viewing environments. e, Structured observations retaining images, text, grounding, trajectories, revisits and interpretations. f, Examples of observations linked to the tissue being examined, including peripheral palisading, myxoid stroma and uncertainty about an infiltrative border. g, Processing of viewing events, speech alignment, gaze fixations and review history. h, Recording duration by tissue or disease group and participant training levels. The current collection contains 48 sessions, 1,028 min and 16 pathologists; n denotes sessions. i, Narration categories by tissue or disease group and in the pooled set; numbers in parentheses denote categorized sentences. Each sentence received one primary LLM-assigned category; 2,879 sentences assigned to “other” were excluded, leaving 4,759. Each pie has equal area; diagnosis, architecture and cellular morphology are annotated when they account for at least 10% of sentences. j, Vocabulary enrichment in each of five skin readers relative to the pooled non-skin records. Cells show log-odds z-scores from segment-level lexeme presence with a 0.5 cell correction; positive and negative values indicate higher and lower usage, respectively. Terms were selected exploratorily. k,l, WSI–report pairs and the spatial and procedural information not specified by those pairs. m, Comparison of seven systems or model families across trainability, personal adaptation, evidence localization, visible reasoning and private deployment. Filled circles indicate support, half-filled circles partial support and crosses lack of availability in the compared configuration (Methods). Visible reasoning denotes inspectable examination steps and associated evidence. PathChat has partial support for evidence localization; the representative pathology foundation models (UNI and CONCH) do not provide this output as standalone encoders. PathCowork supports all five capabilities, with model weights released.

# Artwork on page 5.

## Learning visual grounding from narrated examination

The recordings linked diagnostic language to the scale at which a finding was examined. When mitosis was mentioned, the median viewport magnification was 23× (19 observations), compared with 21× for nuclei and 13× for cytoplasm (Figure 2a). Anatomical structures were generally discussed at lower power: median magnification was 4.0× for glands, 3.0× for dermis, 2.6× for follicles and 1.4× for lymph nodes (103 observations). Although the distributions overlapped, the median magnification for mitosis was approximately 16 times that for lymph nodes, linking each concept to the scale at which pathologists examined it.

Spatial descriptions were associated with different extents of gaze movement (Figure 2b). The median root-mean-square spread of gaze around an utterance was 3.0 mm for “diffuse” and 2.3 mm for “infiltrate”, compared with 1.7 mm for “around”, 1.5 mm for “sheets”, 1.1 mm for “nests” and 0.7 mm for “scattered”. These measurements describe the area explored around the spoken observation. Together, the viewing scale and spatial context provide information that a transcript alone does not specify.

We used observations linked to their corresponding regions in the original slides to supervise visual grounding, a critical task in pathology in which a model is given an image and a specified concept and must localize the region corresponding to that concept. The examples in Figure 2c show how narrated microvascular proliferation and mitosis provide training targets for locating those findings. Supervised fine-tuning improved aggregate grounding in both 1T- and 27B-parameter open-source models (Kimi K2.6 and Qwen 3.6 27B)<sup>22,23</sup> relative to their unmodified counterparts (Figure 2d). The score rose from 19.8% to 51.2% for the 1T model and from 5.6% to 52.2% for the 27B model, gains of 31.4 and 46.6 percentage points, respectively. Both adapted models exceeded the GPT-5.4 reference of 24.4%.

We examined this adaptation more closely using seven melanocytic concepts, with training examples drawn from dermatopathology recordings pooled across participating pathologists. Concept recall, precision and spatial grounding were evaluated separately against Claude Opus 4.7, GPT-5.4 and Gemini 3.1 (Figure 2e). The adapted localizer showed stronger grounding for the dermoepidermal junction, dermal melanocytic nests and pagetoid spread, although recognizing a finding did not ensure accurate localization. The qualitative examples make this distinction visible: PathCowork followed the dermoepidermal junction and marked a localized epidermal region for pagetoid spread, whereas comparison outputs included contours displaced above the tissue or regions covering much of the field (Figure 2f). Additional examples are shown in Supplementary Figures S1 and S2 (Supplementary Note 1).

![](images/df16282e19bf1c20ab8bab7e81360deb617aada35cfd056541fe0d9ef3ece364.jpg)  
Figure 2. Legend on page 8.

Figure 2. Learning visual grounding from recorded behavior. a, Viewport magnification associated with mentions of diagnostic concepts. Dots represent observations, violin widths indicate density, and red lines and labels show medians; n denotes concept-associated segments. Magnification is the objective-equivalent viewport scale from per-slide optics metadata; unreliable rotated viewports were excluded. b, Gaze-density contours associated with six spatial terms, aligned on each utterance’s gaze centroid and pooled by term. Annotations give the median per-utterance root-mean-square (RMS) gaze spread. Gaze was pooled from segments overlapping the utterance and its surrounding 5 s, retaining windows with at least three samples. Nested contours enclose 20%, 50% and 80% of smoothed in-field gaze density from dark to light; scale bars, 2 mm. c, Narrated microvascular proliferation and mitosis, showing recorded gaze and the corresponding findings located by the PathCowork grounder. d, Aggregate grounding performance of Kimi-K2.6 (1T parameters) and Qwen3.6-27B after supervised fine-tuning on PathCowork recordings (purple) and without fine-tuning (grey). Scoring uses Dice on filled polygon masks or equal-width, 50-pixel bands around predicted and reference polylines. Bars show the mean of the organ-level average Dice scores; error bars show the standard deviation across organs. The dashed line denotes GPT-5.4; the annotated difference for the 27B model is in percentage points. e, Concept recall, precision and spatial grounding for the PathCowork-1T localizer, Claude Opus 4.7, GPT-5.4 and Gemini 3.1 across seven melanocytic concepts, with 50 test image–concept pairs per concept (350 pairs in total). Test patients were separate from training patients. Radar axes are normalized separately; perimeter values label the individual scales. Training used pooled dermatopathology recordings. Missed targets and unparseable spatial responses receive zero localization scores and remain in the evaluation denominator. f, Qualitative localization of the dermoepidermal junction and pagetoid spread by the four models, shown as outlines and shaded regions. Ticks and crosses indicate correct and incorrect localization, respectively.

Artwork on page 7.

## PathCowork compiles diagnostic experience into AI agents

As the grounded observations describe what to look for, the accompanying narration also describes how to proceed through an examination. PathCowork uses a diagnostic task to retrieve relevant review histories and organize their observations into a reusable diagnostic procedure (Figure 3a,b,c). For melanoma versus dysplastic nevus, the retrieved material includes examination of the dermoepidermal junction, assessment of junctional growth and the dermal component, and a search for mitoses at higher power. The resulting skill links these steps to the anatomical targets, viewing scales and visual concepts needed to carry them out. Viewer logs or gaze associate the narrated concepts with tissue regions, providing paired images and spatial targets for grounding (Figure 3c). These targets can trace a boundary, such as the dermoepidermal junction, or outline regions, such as dermal melanocytic nests. The agent can then use the learned grounder to locate those concepts on another slide. This construction separates the order and purpose of examination from the model that supplies its visual localization, while retaining their common source in recorded cases.

We selected five diagnostic tasks to address common challenges in cancer diagnosis and draw on the expertise of our collaborating pathologists. The resulting skills covered melanoma versus dysplastic nevus, prostate needle-biopsy assessment, glioblastoma versus lower-grade glioma, colon lymph-node metastasis detection and Nottingham grading of invasive breast carcinoma (Figure 3d). Each procedure specified a sequence appropriate to its task. The lymph-node skill, for example, first inventories the tissue and identifies node boundaries, then directs examination of the capsule, sinuses and node interior before zooming into a detected focus. The breast skill instead organizes examination around tubule formation, nuclear pleomorphism and a mitotic hotspot.

We evaluated these agents alongside Claude Code and Codex using the primary outcome for each task (Figure 3e). Giving each of the five tasks equal weight, primary scores were on average 17.9 percentage points higher for PathCowork than Claude Code and 24.1 points higher than Codex. For melanoma versus dysplastic nevus, PathCowork achieved 80% accuracy across 20 cases comprising 69 whole-slide images, compared with 65% for each comparator. It detected all melanomas in this evaluation, with 60% specificity, whereas both comparators had 40% melanoma sensitivity and 90% specificity (Supplementary Figure S3; Supplementary Note 2). The higher accuracy therefore involved a trade-off between missing melanoma and overcalling dysplastic nevi. For prostate assessment, the cancer-detection score was 95%, compared with 85% (17/20) for Claude Code and 35% for Codex. On the same 30 brain cases, PathCowork achieved 86.7% accuracy (26/30), compared with 60% (18/30) for Claude Code and 70% (21/30) for Codex. PathCowork correctly classified 13/15 cases in each class; Claude Code and Codex correctly classified 3/15 and 6/15 GBM cases, respectively, and all 15 LGG cases. Lymph-node detection recall was 97%, 87.3% and 86%, respectively, and breast grade accuracy was 71.8% (28/39), 43.6% (17/39) and 53.8% (21/39), using the same 39 cases for all three agents. Component-score accuracy and grade-specific recall show where these differences arise (Supplementary Figure S3).

The higher task scores were accompanied by shorter processing times across all five tasks, with 6.2- and 4.7-fold speedups over Claude Code and Codex, respectively, calculated from the task-averaged times (Figure 3e). For skin, PathCowork required a mean of 39 s per slide, compared with 252 s for Claude Code and 117 s for Codex. For brain classification, the corresponding times were 20 s, 332 s and 285 s; for breast grading, they were 106 s, 300 s and 192 s. Colon lymph-node examination took a median of 22 s per slide with PathCowork, compared with 277 s and 322 s for the two comparators.

Reported inference costs were also lower for PathCowork in every task, ranging from \$0.01 to \$0.17 per slide, compared with \$0.47–\$1.98 for Claude Code and \$0.75–\$1.25 for Codex. The ratio of the task-averaged costs corresponded to 11.0- and 10.3-fold reductions, respectively. Skin examination cost \$0.17 per slide with PathCowork, versus \$1.00 and \$0.75, respectively; colon lymph-node examination cost \$0.01, versus \$0.96 and \$1.04. These comparisons evaluate the complete behavior-derived agents: recordings supply the procedural skills, their execution sequences and the supervision used to adapt the grounder.

Model weights for PathCowork-1T (server deployment) and PathCowork-27B (local deployment) are released at https://github.com/zhihuanglab/PathCowork. Both learn visual grounding from the recorded observations and can supply the localization used by the diagnostic procedures. The agent results above use the PathCowork-1T grounder; the paired grounding comparison evaluates both models (Figure 2d). To further leverage the full breadth of pathologist behavior collected across multiple organs and diseases, beyond the five diseases evaluated in our diagnostic benchmarks, we explored General navigation learned from pathologist behavior, implemented as PathCowork-general, which learns generalizable viewing actions to support evidence gathering and subsequent LLM interpretation for openended diagnostic questions not yet covered by a distilled skill (Supplementary Note 3; Supplementary Figure S4; Supplementary Video 3).

Metastasis detectioNo tingham grading

Melanoma vsGl s n grad dysplastic nevus

ProstateB east

a

## PathCowork at pathologists’ backgroundbackgr und

![](images/c3c87eb273dcdc7731a8a5890a7de4f579bd3d56a3bf26f043db5ae267de1a41.jpg)

d  
![](images/398d3f5203abbdadc90d4ff513428f88cdf0691c30536fc4e9728e0e95b0b0f4.jpg)  
b

Can you help distinguish between<sup>PathCowork</sup> <sup>co</sup>een melanoma and a dysplastic nevus?<sub>Anatomy Junc</sub>vus?

## Retrieve PathCowork slide historystory

Task define dysplastic nevus vs. melanoma

## Keyword expansion

dysplastic nevus / nevi atypical melanocytic nevusMelanoc.melanocytic nevus Mouse hov melanoma / malignant melanoma melanocytic neoplasm radial / vertical growth ...ertical growth ...

![](images/5746953322be79f2ac9d67e50124ed21586decc15afbc6b57e83bcd813385a01.jpg)

## Retrieved record · illustrative narration

JUNCTION “At the DEJ, I see pagetoid spread, single melanocytes and confluent nests.”

![](images/685c7d888e2be10ec1f05f0d5731f478ab21fbb82950af0f6ab8883cfcbce7b9.jpg)

![](images/9f8553855442f87ac818611a5f49012f19d34e90567dd944db1c17bc8617e7d6.jpg)  
c

## PathCowork compiles the pipelinepeline <sub>Can</sub> <sub>you</sub> <sub>help</sub> <sub>distinguis</sub>

![](images/6da4e5e9e97d65437b6cb42ad396fcad5a1c9ef0a9619e152dba7bbbca8a7bc0.jpg)

![](images/f7147cffffa71173cba128f2ec09e7c5133fa7c8a03a7f17936350cac74886ef.jpg)  
PathCowork collects the grounding datasetPathCowork collects the grounding datasetunding datasetunding dataset

![](images/64baf3ccce14bb2de57460017ace0b6b595298c7002431c8a0e4447764c69bec.jpg)

## Dysplastic NevusProst te Needle vs MelanomaGleason Gra

## Prostate Needle BiopsyGBM vs Lower-Grade G Gleason GradingH&E Morphologi

## GBM vs Lower-Grade GliomaColon Lymph Node  Dyspla tic N vus H&E Morphologic FeaturesMetastasis Screening vs Melanoma

1. Tissue inventoryInventory & inde 20.0 µm/px4   
Count and index all tissue   
fragments.lymphoid tissu   
tissue fragments · section coverageorientation · i egrity · architectu e <sub>2.</sub> <sub>Survey</sub> <sub>&</sub> <sub>tumor</sub> <sub>map</sub>2. Node grounding   Track the DEJ 8.0 µm/px Map tumor, infiltrative margins and pale zones.   
geographic / serpiginous necrosismac o met tumor d p t e<sup>DEJ</sup>   
Map targets for higher-resolution review.3. Junctional atypia 3. Sample necrosis rimsNo macro met / deposit: 1.0 µm/pxom Sample 2.3 mm fields alongalong the grounded border.  nests confluent   
necrosis boundaries.fol ow grounded capsule / sinus   
<sub>MVP nuclear</sub> <sub>palisades</sub>detect candidate foci along the borde   
No necrosis: sample dense tumor / interfaces.

## Colon Lymph NodeInvasive Breast Ca  Prostat Ne dle Bi Metastasis ScreeningNo tingham G ading Gl ason Grading

## Invasive Breast CarcinomaGBM vs Lower-Gr de Glio Nottingham GradingH&E Morphologic Fe

1. Survey & tubule score Tissue inventory 8.0 µm/px20 Map invasive tumor and assessCount and index all tissue tubule formation.fragments.   
tubules solid nests / cordsiss fragments · section cove age   
>75% → 1 · 10–75% → 2 · <10% → 3

2. Nuclear pleomorphism Map tumor, infiltrative ma 1.0 µm/pxgins and Assess nuclear size, shape and nucleoli.geographi <sub>uniform</sub> <sub>→</sub> <sub>1</sub> <sub>·</sub> <sub>moderate</sub> <sub>→</sub> <sub>2</sub> <sub>·</sub> <sub>marked</sub> <sub>→</sub> <sub>3</sub>Map targets for higher-resolution review. Compare with normal epithelium / lymphocytes.

3. Select mitotic hotspotSample 2.3 mm fields al 0.5 µm/pxg Find the most active, cellularnecrosis boundar es.   
invasive edge.MVP nuclear   
distinct mitotic figures   
avoid necrosis / crush areas   
4. Count mitoses <sub>0.5</sub> <sub>µm/px</sub><sup>0.5</sup> <sup>µm/px</sup>   
Count consecutive fields totaling<sup>res</sup> <sup>at</sup> <sup>hi</sup>   
<sub>2.0</sub> <sub>mm².</sub>resolutio   
<sub>exclude</sub> <sub>apoptosis</sub> <sub>/</sub> <sub>pyknotic</sub> <sub>nuclei</sub><sup>palisades</sup> <sup>bordering</sup> <sup>a</sup> <sup>necrotic</sup> <sup>cent</sup>   
<sub>exclude</sub> <sub>crush</sub> <sub>artifacts</sub> <sub>·</sub> <sub>adjust</sub> <sub>field</sub> <sub>count</sub><sup>Exclude</sup> <sup>folds,</sup> <sup>cautery</sup> <sup>and</sup> <sup>crush</sup> <sup>artifacts.</sup>

## e SkinPros

![](images/cccb21057cd91d64c277beb314b7d31ebccf179d5db37bd57389338b75b5771e.jpg)

![](images/3c993b9879c9a678590ebb5ffb91d8347ef527b8ed6c56ae20def6bc99740bee.jpg)  
Figure 3. Legend on page 11.

BrainColon LNP ostate  
![](images/cd5ae6b857eda75d401a1b8c3a06e1bf5677e825d44127e7d6e4ef942ee76670.jpg)

## Colon Breast

![](images/bf4344f4863b1ee8dd764c1d89d4fe4e85e565b000cbbff67141c2ab3c9a70d1.jpg)

Figure 3. Constructing and evaluating diagnostic agents from recorded experience. a, Collection of multimodal review histories across sessions. b, Retrieval of relevant histories using a diagnostic task and expanded queries, with an illustrative narrated record. c, Compilation of diagnostic steps and collection of visual examples for grounding. The branching schematic represents the examination procedure. Below, paired images illustrate behavioral context from viewer logs or gaze (left) and corresponding spatial targets (right): a polyline for the dermoepidermal junction and polygons for dermal melanocytic nests. Overlays are schematic; stacked cards and ellipses indicate additional examples and concepts. d, Five compiled skills for skin, prostate, brain, colon lymph nodes and breast, in that order. The excerpts specify examination steps, targets and viewing scales. e, Task performance versus cost per slide for PathCowork (purple), Claude Code (orange) and Codex (grey outline). Bubble area represents time per slide; labels give the score and time. Primary metrics are skin diagnostic accuracy, prostate cancer detection, brain diagnostic accuracy, colon node detection recall and breast grade accuracy. Colon time is the median; other displayed times are means. Task-specific cohorts and configurations are detailed in Supplementary Note 2.

## Artwork on page 10.

## PathCowork guides slide-level learning with behavior-derived regions

The adapted agent could also supply spatial guidance for training another model. We tested this with our proposed “CoworkMIL”, which uses regions proposed by PathCowork to guide multiple-instance learning (MIL) for prostate grading (Figure 4a). The agent examined slides in the MIL training set and identified regions of interest. These regions supplied tile-level supervision alongside the slide labels, directing training toward the tissue selected by the grounder. The regions were required during training only; the trained model subsequently predicted grades from the slide images.

We compared CoworkMIL with attention-based MIL (ABMIL)<sup>6</sup> and a CLAM-inspired pseudo-label control<sup>8</sup>, using three representative pathology image encoders (UNI2-h, Virchow2 and CONCH), 10–50 training slides and 20 seeds per setting (Figure ${ \pmb 4 6 , \pmb c , \pmb d , \pmb e , \dag } )$ . At 20 training slides, CoworkMIL increased mean quadratic-weighted κ over ABMIL from 0.466 to 0.544 with UNI2-h, from 0.385 to 0.486 with Virchow2, and from 0.484 to 0.585 with CONCH. It also exceeded the mean performance of the pseudo-label control with each encoder. The largest gain over ABMIL was 0.151 with Virchow2 at 30 slides (κ = 0.620 versus 0.469). The benefit depended on the amount of training data and the encoder: at 50 slides, the gap over ABMIL narrowed to 0.013 with Virchow2 and became negative with UNI2-h, whereas CONCH retained an advantage (κ = 0.741 versus 0.643). Across the tested training-set sizes, gains were most consistent with CONCH.

In the auxiliary-loss sweep, κ increased from 0.500 at weight β = 0.1 to 0.583 at β = 1.0, then changed little at $\beta = 2 . 0 ( 0 . 5 7 6 ;$ Figure 4g). In the illustrated biopsy, CoworkMIL concentrated attention on glandular tissue, whereas the highest-attention ABMIL patches included more surrounding stroma (Figure 4h).

As a separate comparison, EAGLE<sup>24</sup>, which selects tiles with frozen attention and fits a slide-level linear classifier, achieved mean κ of 0.430 and 0.435 at 0.5 and $2 . 0 \mu \mathsf { m }$ per pixel, respectively, using the same 20-slide training subsets and 30-slide test cohort (Supplementary Note 4). These values were below the 0.486 obtained with Virchow2-CoworkMIL, although the pipelines differ in spatial sampling, feature representation and training supervision.

![](images/a1554bcbc896778a26f5d668dc9cf679ea1f4eb42e5e85dd288fae8a7eb1f78d.jpg)

## PathCowork grounds diagnostic patterns

Find prostate growth patterns to guide slide-level grading.

1 Apply behavior-trained concept grounding

![](images/7bfc0262881b8b089a1b7bf4a0152564a3e77e130c084b346e7c39b80916e90f.jpg)

2 Transfer grounding to the tile grid

![](images/dad1489a653f1c19b8a15ad016f536b1e998227f268ec0353ba999fdc154b1f3.jpg)

![](images/a592ea31822918184d15e25d5b840a1389d0470722c7b16b27cbe2615a94ce7c.jpg)  
Learn a slide classifier from grounded tiles

![](images/11efbd409f504ce51d0b384f38fd70b5699a4e51f9af73a2a4c603c46f0dc67b.jpg)

![](images/c09fd7df16ddb620cf7ac837f4d2f2550466c84f03cd27d1a302fbd0754f9727.jpg)

![](images/46f56e591b6b9f66b1a4bd8dd905ccb5e190652bec174cd9a764202bc8915cc5.jpg)

![](images/14f93becfc7f54f2eea6ebe85d0151b3303a0340226295766554fa4d9c1a4706.jpg)  
Spatial supervision during training

![](images/fb1e1cd958937a48b019e792e32be3d4d6b63b8545919527e07723a40d61b851.jpg)  
Inference: image features only · no ROI required

![](images/61d95e876fbc9d450e9a224ef102f452e653526b517be5de2ce41999a66b6498.jpg)

![](images/bba60088c3d5453bb99349792d983f7fd3ef7de2d2b0defc2f17153621092a4b.jpg)

![](images/b3a112744d01ebf4c965834892900edb4edf0c90f53d59840972f44c9c2c0533.jpg)

![](images/7b23809be7f4ad75dd2e524ad55d6291615c5fd95b065dae7cdc97b8f492760c.jpg)

h  
![](images/47ea2c1e68604720d632a0812b317bb956a55e2cd8ea8da6664dfb1678bea51d.jpg)

![](images/804b01c1b65a1d33d4356913bb94784650b01f58843d327e7870443c62ebece9.jpg)  
Top-16 patches of ABMIL

Figure 4. Legend on page 13.  
![](images/134b6b120bcb6d922c27124331b9d442e985ad193857072d2cb3da175b17bfea.jpg)  
Top-16 patches of CoworkMIL

Figure 4. PathCowork-guided multiple-instance learning for prostate grading. a, Expert review histories provide behavioral supervision for the PathCowork grounder, which proposes suspicious regions on multiple-instance learning (MIL) training slides. Region polygons are converted into binary tile targets. A frozen image encoder supplies tile features; learned tile logits receive auxiliary binary supervision and define attention weights for slide-level pooling. Training combines slide-label cross-entropy with tile-level binary cross-entropy, with a fixed auxiliary weight of $\beta = 1$ in $\mathbf { b } , \mathbf { c } , \mathbf { d } , \mathbf { e } , \mathbf { f } .$ PathCowork guidance is used only during training; inference requires the slide alone. Network glyphs and attention weights are schematic. ${ \mathsf { b } } , { \mathsf { c } } , { \mathsf { d } } ,$ , ISUP grading performance (quadratic-weighted κ) versus training-set size for UNI2-h, Virchow2 and CONCH, comparing ABMIL (grey), the CLAM-inspired pseudo-label control (labelled CLAM; blue) and CoworkMIL with agent-generated ROIs (purple). Lines show means over 20 matched training seeds on a fixed 30-slide test set; bands show standard errors calculated across seeds (Methods). Stars denote paired, two-sided exact Wilcoxon signed-rank tests against ABMIL: \*P < 0.05, $^ { \star \star } P < 0 . 0 1$ . Exact P values are listed in Supplementary Table S5. e, Difference in κ between CoworkMIL and ABMIL across training-set sizes for each backbone. f, Comparison of the three methods at 20 training slides; bars show means over 20 seeds, and error bars use the same standard-error calculation as in ${ \begin{array} { r l } { \mathbf { b } , \mathbf { c } , \mathbf { d } . } \end{array} }$ g, Effect of auxiliary-loss weight β on CoworkMIL performance (UNI2-h, 20 training slides), averaged over ten seeds. These are the first ten seeds of the main comparison; the dashed line denotes their ABMIL mean. h, Illustrative prostate biopsy with CoworkMIL and ABMIL attention maps and the 16 highest-attention patches from each model.

Artwork on page 12.

## PathCowork builds pathology AI benchmarks from recorded clinical practice

Beyond training AI models, PathCowork also enables the construction of case-specific benchmarks from recorded clinica examinations, allowing us to evaluate whether AI models can understand the visual context, interpret evidence, and make decisions encountered during real-world pathology practice. Specifically, we use PathCowork to generate case-specific questions from the pathologist’s observations and viewing history, covering four demands: integrating evidence across slides, rejecting a question with an inapplicable premise, identifying the tissue location or viewing magnification, and choosing where to look next for a stated goal (Figure 5a). We formally named this benchmark as “CoworkExam”. For example, a recorded recommendation to request C-kit staining supplies the answer to a question about which immunostain to order after reviewing several slides. Other questions require recognizing that a field does not contain the tissue assumed by the question or selecting a region for closer examination (Supplementary Figure S5). A separate text-based analysis examined how much of the recorded checking content was retained by generated questions, compared with questions selected from PathMMU and textbook-derived banks (Supplementary Figure S6; Supplementary Note 5).

We compared questions drawn from recorded cases with questions from PathMMU and a dermatopathology textbook (Figure 5b; Supplementary Note 5). On the same 148 CoworkExam questions, Opus 4.8 answered 39 correctly without images (26.4%) and 121 with images (81.8%), a difference of 55.4 percentage points. The displayed PathMMU scores were 59% and 74%, respectively (15 points). The displayed GPT scores were 23% and 55% on CoworkExam (32 points), versus 62% and 70% on PathMMU (8 points). Image-supported accuracy was higher for both question sources, with larger source-level differences on CoworkExam.

A separate comparison across Haiku, Sonnet and Opus showed a wider range of scores on CoworkExam questions: 41%, 52% and 77%, respectively, versus 56%, 73% and 74% on PathMMU and 32%, 30% and 33% on textbook questions (Figure 5b). The Sonnet-to-Opus difference was 25 percentage points on CoworkExam and 1 point on PathMMU. Recorded-review questions yielded a wider spread of scores across these model configurations.

Across the four categories, six frontier models and one pathology resident showed distinct capability profiles (Figure 5c). The resident scored 91% on rejecting inapplicable premises, compared with 13–79% for the models, and 84% on environmental sensing, compared with 46.5–79%. By contrast, the highest model scores were close to the resident’s on multi-slide reasoning (89% versus 87%) and goal-directed navigation (90% versus 89%). The largest remaining gaps were in recognizing the visual context of an examination and identifying when a question was based on an incorrect premise, highlighting the need for more rigorous pathology AI benchmarks and better reasoning models that can learn from real-world clinical practice, as enabled by PathCowork.

Recorded reviews

Opus 4.8 Haiku 4.5 GPT-5.5 GPT-5.6-sol GPT-5.4-mini Gemini 3.1 Resident

![](images/551fa8faa8b21e7b9a26b054753abc624b47a6adde144fbc14962194f665c5aa.jpg)

![](images/d340ecc50ae00d2025fb82913769ea5ad8c8647f910ef8214c5176a5a8031257.jpg)

b  
![](images/ee0e2b6ff4e33f414002438fa60ce26efffee7085eb42af0ddaac754c8f62496.jpg)

![](images/03b4f284ca99a9ed4e866841aa43dfa107ab6da43c1a1a0c1bf9ac0041777bd5.jpg)

![](images/22e7f74397fa8584b7a138f306d8a1545b9ee9cd81f042cb4fb048807583d72d.jpg)  
Published cases

Public questions

## Long Context Multi-Slide Questions

![](images/66c668ecbfe880c8723f3e755821a93b8bae6d779b87fa11110421f16183f0bb.jpg)  
Q: Pathologists want one immunostain to settle the case. Which IHC stain should be ordered?

(A) SOX10

(B) CD68

(C) CD117 ✅

(D) TTF-1

![](images/35674e0e43f69357e182b230872e3eb95eca6afc9ec3c0488bf6367cae0d204d.jpg)  
Ground truth from: "So I'd say get a C-kit,"

![](images/15059b32c9c189af64264898d8dc46264a1349c4b3f764c864a3785636eab12f.jpg)  
Teaching cases

Assembled questions

![](images/1967a736c6a43c34000d7a7da9ffbdab3b1ba02239d1beb9e37cf28085427441.jpg)

![](images/fa25cbe6bae50b0f0343d88c8f538ea7c734c68b1d21c8409cd810552c63a2c4.jpg)

c  
![](images/44a3655fc382686de1bcfba3a71618af28ae07157a05fedb41f22a1cef56591a.jpg)

## Hallucination

![](images/a63d9847502685e26fe1c61a3ec0088862ce705e60f22fb9c8968c4bf5f1c3e0.jpg)

Q: Does the melanoma infiltrate the DEJ? (A)Yes, the melanoma...

## Env. Sensing

![](images/678ba15422c86124efd38644c5e92470e1f8285aa8f1939fc9c0c7bc25b272bf.jpg)  
Q: Where the specimen is this field taken from?

(D) does not apply ✅

![](images/1b9860f3a9e287bc70ce738e3b8cf2a618103842bd3b3a9c3de3a5f1de690eb4.jpg)  
Q: Roughly what mag. are you viewing at?

![](images/3fb215c2eeafe00def7ade42a70d83259018ac9572125cf4cd8e6e0026bfcdd4.jpg)

![](images/f364e4eb821b57086c5084b36d75bb80d84fae2bed786e1a15e4eb258b61c2e6.jpg)  
Figure 5. Legend on page 15.

## Goal-directed Navigation

![](images/9e70b63aacdf182397da8214e46584d2b7cb9159c8620404ac2fa55a60bc05a8.jpg)  
Q: Given the goal to examine the tumor morphology to determine its histologic type, which region should you examine next at higher power?

![](images/e4e621362c3bf9aa6a40fe94df3f32167043af770625fb4a1a9fd8647a9ec19b.jpg)  
Environmental Sensing

![](images/7be97a4dae04eb80a6130d113bd81765e5afd0a6c4fea98500ce5590dfd4ca1d.jpg)

Figure 5. Evaluating models using the context of recorded examinations. a, Generation of case-specific questions from review histories, with examples of multi-slide reasoning, rejection of an inapplicable premise, environmental sensing and goal-directed navigation. b, Left, schematic comparison of question sources. Middle, displayed accuracy without images (lower bar segment) and the additional score with images (upper segment) for Claude and GPT on PathMMU and PathCowork (CoworkExam); the total bar height gives accuracy with images and the dashed line marks the 25% reference for a four-option item with one accepted answer. Segment annotations give percentage-point differences. The Opus 4.8 comparison on CoworkExam uses the same 148 questions in both input conditions (39/148 without images; 121/148 with images). Right, accuracy on the separate 100-item sets per source across Haiku, Sonnet and Opus on PathMMU, PathCowork and textbook questions; the horizontal axis shows the input-price values used in the source comparison. The source schematic shows the provenance of the question materials. c, Accuracy of six frontier models and one pathology resident across four question categories: rejection of an inapplicable premise, multi-slide reasoning, goal-directed navigation and environmental sensing.

## Artwork on page 14.

## Discussion

Clinical AI has largely learned from static images and final diagnostic reports, leaving the interpretive process that connects them unrecorded. PathCowork addresses this gap by capturing how pathologists examine tissue, link spoken observations to specific image regions and viewing scales, and build diagnostic explanations. Treating clinical interpretation itself as the primary source of supervision marks a fundamental change in how medical experience becomes AI learning, moving beyond predicting endpoint labels to capturing how experts reason through visual evidence during diagnosis.

This work has several strengths. Most fundamentally, it transforms routine diagnostic experience into durable, reusable knowledge, preserving the reasoning that normally vanishes once a report is signed. We developed an evidencelocalization grounder that outperformed frontier commercial models. Agents combining this grounder with recorded diagnostic procedures were more accurate, faster and substantially cheaper to run than Codex and Claude Code. Furthermore, the value of this captured expertise extends beyond agents: using agent-identified regions to guide slidelevel model training when labelled slides are limited shows that recorded clinical reasoning can become transferable supervision across computational tasks. Finally, this shift from static answers to active reasoning changes how medical A should be judged. Through CoworkExam, evaluation moves beyond factual recall or disease naming to test interpretation in context, including whether a model can recognize the viewing context, reject inapplicable premises, integrate evidence across slides and decide what to examine next.

This work has several limitations. First, the current evaluations measure the standalone diagnostic accuracy of PathCowork, establishing the performance of the agent alone. In the foreseeable future, however, PathCowork is intended to serve as a copilot for pathologists, and how the system affects clinicians during routine diagnosis remains unstudied. Prospective studies of real clinical use, although out of the scope in this work, will be needed to determine whether pathologists using PathCowork work faster, diagnose more accurately, gather more complete evidence and experience less subjective fatigue over the same duration of work, while also assessing their perceived usability and overall experience with the system. Longitudinal follow-up should also examine whether sustained use maintains or erodes clinicians’ unaided diagnostic proficiency.

Second, the current recordings capture the complete traditional diagnostic workflow of a pathologist directly examining tissue images and interpreting findings. Increasingly, however, pathologists are adopting AI tools, such as automated cell counting or mitosis detection, to accelerate their work. Extending PathCowork from two-way interactions between pathologists and images to three-way interactions among pathologists, images and AI tools would capture how clinicians use and interpret AI outputs during diagnosis, enabling future agents to learn from these interactions.

Third, learning from recorded examinations is currently passive, so diagnostic concepts may receive uneven supervision depending on how often they arise during routine cases. A partly active learning approach could allow an agent to recognize concepts or subtle morphological distinctions it has not yet learned well and ask targeted questions, request demonstrations or seek corrections during an examination, focusing supervision where needed and adapting the system to an individual pathologist’s specialized expertise.

Looking ahead, these capabilities point toward AI systems that could shadow pathologists during routine diagnosis, learning mostly by observation while asking targeted questions when unfamiliar or ambiguous findings arise. Paired with rigorous benchmarks such as CoworkExam to test whether an agent has learned to interpret visual evidence in context and act appropriately, this direction offers a foundation for transparent, personalized clinical copilots shaped directly by the experts they work alongside.

## Methods

## Study design and recorded corpus

The study was approved by the Institutional Review Board of the Perelman School of Medicine at the University of Pennsylvania (IRB #857258), and all participants provided informed consent. Recording took place from February 2025 to September 2026. The recorded corpus comprised 48 sessions and 1,028 min of review from 16 pathologists: six attending physicians, four fellows and six residents. These 16 participants included 12 co-authors of this study and four pathologists whose recordings were reused from our previous Pathology-CoT study<sup>20</sup>. Participants reviewed diagnostic or teaching material while explaining their observations as they would during a discussion at a microscope. Eye-tracker calibration preceded recording. Recordings with unusable audio were excluded from transcription-based processing.

A session denotes one continuous sitting at the WSI viewer; it can contain several slides, and a slide can be revisited. Sessions, slides and extracted viewing segments therefore represent different units of observation. The image collections span skin, prostate, breast, brain, lymphoid, gastrointestinal, bladder and cytology material (Supplementary Table S1). Public recording sources included The Cancer Genome Atlas (TCGA)<sup>25</sup>, DROID-skin<sup>26,27</sup> and LNCO2<sup>28</sup>. Recordings were processed after the session to produce learning material.

The recordings supplied grounding-training examples, grounding-evaluation examples and material for skill extraction. Grounding training and evaluation were separated at the patient level. Skills and complete agents were evaluated on unseen cases outside the recordings used for grounding training and skill construction. Benchmark construction drew on the full recorded corpus; benchmark evaluation therefore included external models and a pathology resident, without evaluating the PathCowork models.

## Recording platform and synchronization

PathCowork records slide review and narration on a common timeline. The platform extends the TissueLab whole-slide image (WSI) viewer<sup>21</sup>, with a Next.js browser interface and an Electron desktop application. Viewport coordinates, magnification, mouse position, gaze and audio are recorded while a pathologist examines a slide and explains the findings.

Gaze coordinates enter the viewer through a WebSocket connection to a calibrated eye tracker. Mouse and viewport events are captured through viewer hooks and browser event listeners, and audio through the media-capture interface. Mouse and viewport state are polled at 10 Hz, and the gaze buffer is read at 20 Hz; the tracker sampling frequency is distinct from this readout rate. Events are timestamped against the workstation clock to align speech with the visual field. Alignment resolution depends on polling, buffering and transcription timing.

Sessions were recorded on Windows workstations with the available eye-tracker drivers. The software supports local and server-backed storage. Screen recordings allow the event streams to be checked against the slide displayed during review. Subsequent processing operates on the saved streams.

## Processing viewing events and speech

The event-processing pipeline groups pan, zoom and dwell records into viewing episodes. Low- and intermediate-power events capture tissue survey, whereas high-power events capture examination of individual cellular features. Survey views are binned to 5× or 10× equivalents. A stationary view lasting more than 1 s, or continuous panning lasting more than 2 s at these scales, defines an inspection event. High-power views yield a 1,024×1,024-pixel crop centered on the viewport. Brief navigation events are retained in the trajectory.

Audio is transcribed with Whisper-large-v3-turbo<sup>29</sup> using pathology vocabulary and slide context as prompts. Segment timestamps link each utterance to the corresponding viewing episode. A Qwen3-4B text classifier<sup>23</sup> receives the utterance together with two preceding and two following segments to distinguish diagnostic observations from interface discussion, off-topic speech and other non-diagnostic content. The raw transcript is retained alongside the normalized description for checking against the original speech.

## Gaze filtering and spatial association

Gaze samples are mapped from screen coordinates into WSI coordinates using the recorded viewport. A hybrid velocityand dispersion-based procedure identifies fixations<sup>30</sup>. The velocity stage uses a threshold of 1,800 pixels s<sup>−1</sup> in the mapped image coordinates. The dispersion stage uses a 220-pixel threshold and a minimum duration of 90 ms, with dispersion calculated as the sum of horizontal and vertical coordinate ranges. Each retained fixation carries its start and end times, duration, centroid, sample count and dispersion.

For each speech-aligned region, retained samples are rasterized into a two-dimensional density map, smoothed and thresholded to obtain contiguous attention regions. These regions can be represented as boundaries or polygons in WS coordinates (Methods Figure M1). These boundaries represent the clinician’s sampled visual attention.

Processed Boundary

05 seg\_0012   
A blood vessel containing erythrocytes is visible   
within the hypercellular   
tumor, representing a   
component of microvascular proliferation.

Initial Gaze

10 seg\_0018 So this is a little focus of microvascular proliferation.

25 seg\_0038   
you'd find some normal, like   
the remainders of the   
processes 31 seg\_0049   
And some of these may not be neoplastic.

Processed Gaze  
![](images/44e167cfb08a578c08616d9b3a34e6c01dc300a2b7b89e7b45fe3f2d990cc3c3.jpg)

46 seg\_0069 see lots of proliferation. Let's see if we see it. Yeah, so here's a mitosis You can see that

Methods Figure M1. Gaze processing and extraction of attention regions. Each row shows a recorded segment and its aligned narration. Columns show the initial gaze samples, samples retained after velocity- and dispersion-based fixation filtering, and the resulting attention boundaries over the viewed tissue. The side-by-side views allow the extracted regions to be inspected against the tissue and narrated findings, including microvascular proliferation and mitosis in glioblastoma.

## Cross-modal association and review

The pipeline forms candidate regions from gaze fixations, mouse-position density and viewport changes concurrent with speech. An utterance can have fewer than three usable candidates, depending on the recorded behavior. Qwen3.5- 397B-A17B<sup>23</sup> receives the image, candidate overlays and aligned narration and selects the region that best matches the described finding. The selected geometry, source modality and narration form a localized training example.

Nearby viewing events at similar magnifications are merged to simplify the navigation trajectory. Returns to a previously examined region are retained as revisits. A review interface presents the crop, transcript, generated description and preceding context together. Reviewers can check their correspondence and export decisions for downstream filtering (Methods Figure M2).

![](images/cf2e577c47443298f6691c0c75180a6bc3245e86b23cac3d5ca8a129b4d9df68.jpg)

3/109- Segment 0010id=human\_0010 type=human\_readablestatus=unreviewed

So here now we're getting the classic features.

This high-power view demonstrates classic histological features of glioblastoma, including high cellularity and nuclear pleomorphism. A prominent area of microvascular proliferation is visible, characterized by thickened, multilayered vessel walls.

![](images/2af50b824761241493faf7a508c60128ec4555ce43f52ec92c33afb58eb9414b.jpg)

Methods Figure M2. Interface for reviewing generated examples. The reviewer displays the original utterance, the generated description, preceding narration and the corresponding image. Controls record approval or rejection and export the review decisions. The screenshot illustrates an unreviewed example.

Each processed example retains its source image, original narration, timestamps, WSI coordinates and viewing scale. Generated descriptions and contextual summaries accompany the recorded speech; fields for uncertainty and interpretation are inferred from that narration. Trajectories preserve the sequence of views and link repeated visits. These links make derived examples traceable to the original slide and utterance. Detailed fields are listed in Supplementary Table S2.

## Descriptive analysis of recorded behavior

For the narration composition analysis, an LLM assigned each sentence one primary category using the preceding sentence as context. Among diagnostic-review recordings, 7,638 sentences were categorized; 2,879 assigned to “other”

were excluded, leaving 4,759 pathology-related sentences. This categorization is separate from the diagnostic-content filter used during record processing. Vocabulary enrichment was calculated from segment-level presence or absence of each lexeme for each skin reader versus the pooled non-skin records. The log odds ratio used a 0.5 correction in all four contingency-table cells and was divided by its estimated standard error. Displayed terms were selected exploratorily for consistent directions across readers.

Concepts were identified by lexical matching and could overlap within a segment. Objective-equivalent viewport magnification was calculated as the slide’s base objective multiplied by $\sqrt { ( 2 0 4 7 \times 1 4 4 0 ) / ( w h ) }$ , where w and h are the viewport dimensions in slide pixels. Per-slide optics metadata supplied the base objective; unreliable rotated viewport boxes were excluded. For the spatial-term analysis, gaze points were pooled from whole viewing segments overlapping an utterance and its surrounding 5 s. Windows containing fewer than three points were excluded. RMS spread was the square root of the mean squared distance from the window centroid, converted to millimeters using per-slide pixel spacing. Centered windows were pooled by term with equal weight per gaze sample, binned within ±6 mm and smoothed with a $3 0 0 \mu \mathsf { m }$ Gaussian. Contours enclose 20%, 50% and 80% of the smoothed in-field mass. These analyses are descriptive: segments and overlapping gaze windows within a reader are not independent observations.

## Construction of executable skills

Skill construction begins with a target clinical task. Task names are expanded into related diagnostic terms and used to retrieve relevant recorded narration and case context. For the melanoma-versus-dysplastic-nevus task, these terms include atypical nevus, melanocytic neoplasm and melanocytic proliferation. The retrieved material is pooled across the participating dermatopathologists.

Narration is organized into an ordered procedure, such as finding the dermoepidermal junction, examining junctional growth, assessing the dermal component and searching for mitoses. Each procedural concept is then linked to supporting image regions and the magnifications at which those regions were examined. The available spatial representation can be a polygon, bounding box or polyline, depending on the finding. Repeated retrieval and grounding connect the procedural steps to examples from the recordings.

The resulting skill.md specifies the target anatomy, viewing scale, action and decision context for each step. The execution sequence implements the examination order and viewing scales extracted from the recordings. The skill guides an agent that can navigate a WSI and invoke a trained localizer; supervised fine-tuning on the associated behaviora examples teaches the localizer to identify the concepts required by the procedure.

## Supervised concept grounding

The main grounding comparison evaluates Kimi K2 $. 6 ^ { 2 2 }$ and $\mathsf { Q w e n 3 . 6 { - 2 7 } B ^ { 2 3 } }$ before and after supervised fine-tuning on recorded examples. The adapted models are named PathCowork-1T and PathCowork-27B, respectively. Each example pairs an image and concept query with a structured spatial response. The detailed dermatopathology grounding evaluation in Figure ${ \tt 2 e , f }$ and the agent in Figure 3 use the PathCowork-1T localizer. Additional Qwen development examples retain their own model configuration (Supplementary Note 1); the atlas-derived glioma adaptation study below is a separate experiment. Model identifiers are reported by experiment in Supplementary Table S3.

The two main adaptation runs used rank- $\boldsymbol { \cdot 3 2 \mathsf { L o R A } ^ { 3 1 } }$ , a peak learning rate of $1 0 ^ { - 4 }$ , batch size 16 and two training epochs. We used the checkpoint saved at the end of training. Separate development experiments examined model size, numerica precision and training duration to characterize the trade-offs between localization and deployment requirements. The 4B experiments examine spatial-format and reinforcement-learning choices at a smaller scale.

Training examples and reference geometries were derived from recorded viewing behavior and aligned narration, with gaze-derived candidate regions extracted from filtered fixations (Methods Figure M1). Gemini 3.1 screened candidate references for obvious errors, followed by manual spot-checking by a pathologist. Erroneous examples were removed or corrected. Data augmentation and automatic generation of negative examples supplemented the behavior-derived examples. Test examples were constructed using the same procedure, with 50 image–concept pairs per evaluated concept and patient-level separation from the training data. The seven-concept dermatopathology evaluation therefore comprised 350 image–concept pairs (Supplementary Note 1).

In the main grounding evaluation, missed targets and unparseable spatial responses receive a localization score of zero and remain in the evaluation denominator. For the aggregate grounding evaluation in Figure 2d, spatial agreement is measured by Dice, $2 | P \cap R | / ( | P | + | R | )$ , where P and R are predicted and reference masks. Polygons are rasterized as filled masks. Predicted and reference polylines are each expanded into a band of the same 50-pixel width before mask comparison. Dice scores are first averaged within each organ. The aggregate score is the mean of these organ-level averages, and error bars show their standard deviation. Detection recall is the number of correctly detected positive examples divided by the number of reference-positive examples; precision is the number of correctly detected positive examples divided by the number of predicted-positive examples.

Concept evaluations distinguish detection from spatial agreement with the reference region. We report detection recal and precision separately from grounding against regions and contours derived from pathologist behavior. Additional prostate analyses measure localization IoU and the proportion of outputs that can be parsed. We varied training-set size to examine overall and concept-specific detection and localization, and compared fixed-step with equal-epoch training budgets (Supplementary Figures S7, S8 and S9; Supplementary Note 6). Comparisons with frontier models include aggregate prostate results and individual fields (Supplementary Figures S10, S11 and S12). Withholding cribriform examples assessed transfer to a concept without direct supervision (Supplementary Figures S13 and S14). Separate configuration comparisons assessed backbone size, numerical precision and training duration (Supplementary Figure $\$ 15a,b,4,$ Supplementary Note 7).

## Spatial-format selection and glioma adaptation

Before constructing supervised targets, we sampled the unadapted Qwen3.5-4B model under six prompts requesting different spatial formats. The comparison used 60 Ivy $\mathsf { G A P } ^ { 3 2 , 3 3 }$ microvascular-proliferation tiles from eight held-out donors, including 40 positive tiles, at approximately $3 \mu \mathsf { m }$ per pixel and 2.3 mm field width. These tiles were used for format development. Selection considered region production, presence discrimination and whether polygons could represent more than an axis-aligned rectangle. A region was classified as non-rectangular if it had more than four vertices or more than two distinct values along either coordinate axis. Localization was measured separately. The selected prompt contained one hypothetical JSON example; detailed format comparisons are reported in Supplementary Note 8.

Supervised inputs consisted of an image and a concept query, with a target JSON response containing polygons in normalized integer coordinates from 0 to 999, or an absent response for a negative example. Cross-entropy loss was computed only at answer-token positions; image and prompt positions had zero loss weight. We used rank-32 LoRA adaptation<sup>31</sup>, Adam with learning rate $1 0 ^ { - 4 }$ , batch size eight and 768-pixel image inputs. Checkpoints were selected by validation negative log-likelihood using a donor-grouped split. The supervised initialization used for reinforcement learning was trained for two epochs.

Reinforcement learning was run from the base model and from the supervised checkpoint using the same pool of 2,553 records, including 426 negatives. A third condition used a separately supervised checkpoint and a pool augmented with 284 lower-grade glioma negatives. The corresponding supervised validation pools contained 77 records without augmentation and 117 with augmentation. The third condition changes both initialization and the reinforcement-learning prompt pool.

We used grouped response sampling following $\mathsf { G R P O } ^ { 3 4 }$ . At each of 100 reinforcement-learning steps, current weights were re-exported for sampling eight responses to each of 12 prompts at temperature 0.9. Each response’s advantage was the unscaled difference between its reward and the mean reward of the eight responses to that prompt. Groups with identical rewards contributed no gradient. Each sampled batch received one importance-weighted policy-gradient update on response tokens, using rank-32 LoRA and learning rate $1 0 ^ { - 5 }$ . The objective was an unclipped, unregularized group-relative REINFORCE estimator<sup>35</sup>, with the group mean as its baseline and spatial agreement as its reward.

For positive tiles, the reward was $0 . 0 5 I _ { \mathrm { p a r s e } } + 0 . 1 0 I _ { \mathrm { s h a p e } } + 0 . 8 5 F _ { 1 }$ , where the indicators denote a parseable response and at least one non-rectangular polygon. The spatial score $F _ { 1 } = 2 P C / ( P + C )$ combines purity $P ,$ the fraction of predicted area overlapping the atlas annotation, and coverage $C ,$ , the fraction of annotated area recovered; it is zero when there is no overlap. For negative tiles, the reward was $0 . 0 5 I _ { \mathrm { p a r s e } } + 0 . 8 5 I _ { \mathrm { a b s e n t } }$ . Masks were rasterized at 256 × 256 pixels. Positive and negative tiles therefore had reward maxima of 1.00 and 0.90, respectively. Validation localization and presence discrimination were monitored alongside sampled training reward; their trajectories and the format-development results are reported in Supplementary Note 8.

## Scope of the system capability comparison

Figure 1m compares the available implementations of PathCowork, Pathology-CoT<sup>20</sup>, SlideChat<sup>15</sup>, PathChat<sup>14,36</sup>, representative pathology encoders (UNI and CONCH)<sup>9,12</sup>, and the Claude Code and Codex configurations used in this study. PathChat and PathChat 2 are grouped as one family; the SlideChat entry refers to its released CVPR 2025 implementation.

Trainability denotes access to adapting the diagnostic model with additional training data; personal adaptation denotes a mechanism for learning from a clinician’s recorded practice. Evidence localization denotes explicit spatial outputs such as regions, coordinates or annotations. Visible reasoning denotes inspectable examination steps, actions and associated evidence. Private deployment requires patient-facing inference to remain within the user’s infrastructure. Partial support denotes a capability available in only part of the workflow or with limited output. The entries describe implementation capabilities. The dermatopathology application used recordings pooled across participants.

## Agent-system evaluation

The organ-level comparison evaluates PathCowork, Claude Code and Codex on five slide-review tasks: melanoma versus dysplastic nevus in skin biopsies (20 cases, 69 WSIs), prostate cancer detection and ISUP grading<sup>37</sup> (20 slides), Nottingham grading in breast resections (39 matched cases), glioblastoma versus lower-grade glioma in brain resections (30 cases, 15 per class), and metastasis detection in colon-resection lymph nodes (77 slides; ten positive and 67 negative). Evaluation images came from HISTAI<sup>38</sup> (skin and breast), PANDA<sup>39</sup> (prostate), TCGA<sup>25</sup> (brain) and LNCO2<sup>28</sup> (colorectal lymph nodes); evaluation units are listed in Supplementary Table S4. Agents inspect WSIs through slide-navigation tools and return task-specific diagnostic outputs. The PathCowork agent additionally uses a skill assembled from the recordings and a trained localizer.

PathCowork was run once per evaluation unit, defined as a case or WSI according to the task. Its scripted harness executes a finite set of spatial inspection steps at specified viewing scales, calling the grounder to identify the structures required by the skill. For example, junctional examination traverses the identified dermoepidermal junction at the prescribed scale. Execution terminates when the prescribed inspection steps have been completed; no additional agent-turn cap or task-level timeout was imposed on this workflow. An LLM summarizes the collected findings into the final diagnostic output. This completion rule differs from the iterative navigation used by the general-purpose agents.

The brain comparison used the same 30 cases for all three agents, with 15 cases per class. The two missing Claude Code reports were counted as incorrect in the 30-case accuracy and the corresponding class-recall denominator; PathCowork and Codex returned all 30 reports. Runs that did not complete within the time limit were counted as incorrect for task success. Elapsed time and API costs include failed runs through termination; a timed-out run contributes its full time limit and the API cost incurred during that run.

For the 20-slide prostate comparison, a Claude Code timeout is counted as a cancer-detection failure (17/20 detected).   
Its ISUP κ and MAE are calculated from the 19 valid grade predictions, without imputing a grade for the timeout.

Colon node-level references were taken from the LNCO2 ground truth. Node detection recall measures the proportion of reference lymph nodes located by the agent. End-to-end per-node metastasis recall uses all ground-truth tumour nodes as its denominator and counts each node called positive as a true positive; missed tumour nodes count as false negatives. Per-node specificity measures the proportion of reference non-tumour nodes called negative. Slide-level detection is evaluated separately.

The overview uses one primary endpoint per organ: diagnostic accuracy for skin and brain, cancer detection for prostate, grade accuracy for breast, and node detection recall for colon lymph nodes. Secondary endpoints include melanoma recall and specificity, ISUP quadratic-weighted κ and mean absolute error, breast component-score accuracy and grade-specific recall, class-specific brain recall, and colon per-slide recall and specificity. Detailed results, model configurations, task specific denominators and reporting completeness are given in Supplementary Note 2 and Supplementary Figure S3. For breast, all three agents were evaluated on the same 39 cases (10 grade 1, 14 grade 2 and 15 grade 3). Overall-grade and component-score accuracies use 39 as the denominator. Grade-specific recall uses the number of reference cases of that grade. Costs are reported per slide at the recorded API prices; times are wall-clock values, with median time used for colon.

Cross-task summaries give each of the five tasks equal weight. The average performance gain is the arithmetic mean of the five task-specific percentage-point differences between PathCowork and each comparator. For cost and time, we first average the five reported task-level per-slide values for each system, then divide the comparator’s average by the PathCowork average to obtain the cost-reduction factor or speedup. The time summary combines the reported mean times for four tasks with the median time for colon. Calculations use the source values before rounding for display.

## ROI-supervised multiple-instance learning

CoworkMIL transfers regions proposed by PathCowork on MIL training slides into auxiliary spatial supervision for a slide classifier. These regions are outputs of the concept grounder for prostate architectural patterns, including well-formed glands, fused glands, cribriform architecture, solid sheets and cords. Concept-specific regions are combined into binary spatial targets. The image encoder remains frozen while the pooling model learns from slide grades and binary region-ofinterest (ROI) targets. At inference, the classifier uses image features alone, without running PathCowork or supplying ROIs.

From a 100-slide PANDA subset<sup>39</sup>, we constructed a 65-slide training pool and a fixed 30-slide test set, with five test slides in each of six categories (benign and ISUP grade groups 1–5). Training and test slide identifiers were disjoint. Cancer slides without available ROI annotations were preferentially allocated to the test set; cancer training slides required ROIs, whereas benign slides used empty masks. Subsets of 10, 20, 30, 40 or 50 training slides were sampled by cycling through shuffled grade-specific lists. Each comparison used 20 matched runs, sharing the training subset across methods within a run. Subset sampling used seeds 1000–1019, and model initialization and training order used seeds 0–19.

Non-overlapping 256-pixel level-0 tiles were retained at > 25% tissue coverage, estimated by saturation-channel Otsu thresholding on slide thumbnails. Tiles were resized to 224 pixels with ImageNet normalization for UNI2-h and Virchow2, or to 448 pixels by bicubic interpolation with CLIP normalization for CONCH. Cached features comprised 1,536-dimensional $\mathsf { U N } \mathsf { l } 2 \mathsf { - } \mathsf { h } ^ { 4 0 }$ class tokens; 2,560-dimensional Virchow2<sup>41</sup> embeddings concatenating the class token and mean patch token, excluding register tokens; and 512-dimensional pooled $C O N C H ^ { 1 2 }$ image features without contrastive projection or unit-length normalization. Binary tile targets indicated whether each tile center lay within the union of ROI polygons; polygon grade labels were ignored. Outside-ROI targets denote tissue unselected by the grounder.

A linear projection, ReLU and dropout (probability 0.25) mapped each tile embedding to 256 dimensions. A scalar linear suspicion head supplied both the tile-level prediction and the attention logit $s _ { i }$ . Softmax across tiles produced pooling weights, and a linear classifier mapped the weighted embedding sum to six slide categories. For slide label y, predicted class probability $p _ { y }$ and binary targets $r _ { i }$ for m tiles, the effective objective was

$$
\mathcal { L } = - \log p _ { y } - \frac { \beta } { m } \sum _ { i = 1 } ^ { m } \left[ r _ { i } \log \sigma ( s _ { i } ) + ( 1 - r _ { i } ) \log \left( 1 - \sigma ( s _ { i } ) \right) \right] ,\tag{1}
$$

where $\sigma$ is the sigmoid function. The auxiliary binary cross-entropy averages all tiles within a slide, with $\beta = 1$ throughout each main-comparison run. Slide-level cross-entropy was unweighted. AdamW used a learning rate and weight decay of $1 0 ^ { - 4 }$ . Models were trained for 40 epochs with shuffled slide order and one slide bag per update. Evaluation used the final-epoch checkpoint with dropout disabled and argmax prediction.

Comparators used the same frozen features, projection, slide head, optimization settings and training subsets. ABMIL<sup>6</sup> used gated attention pooling and slide-level cross-entropy alone. A CLAM-inspired pseudo-label control<sup>8</sup> added binary supervision to the k highest- and k lowest-attention tiles, where $k = \operatorname* { m a x } ( 1 , \operatorname* { m i n } ( 8 , \lfloor m / 2 \rfloor ) )$ . Its auxiliary loss was $\beta / 2$ times the sum of the positive- and negative-group mean binary cross-entropies. This reimplementation used one shared binary instance head. CoworkMIL used the supervised tile logit for pooling as well as region prediction. The exploratory β sensitivity analysis used the first ten matched runs at 20 training slides with UNI2-h, evaluating weights 0.1, 0.25, 0.5, 1 and 2 on the same test set. Exact paired comparisons against ABMIL are listed in Supplementary Table S5.

We separately evaluated $\mathsf { E A G L E } ^ { 2 4 }$ , a frozen attention-guided tile-selection pipeline, on the same 30 test slides and the same 20 training subsets of 20 slides. EAGLE used frozen CTransPath features<sup>42</sup> and CHIEF attention<sup>43</sup> to select up to 25 tiles, then averaged their 1,280-dimensional Virchow2 class-token embeddings. All feature extraction within a run used either 0.5 or $2 . 0 \mu \mathsf { m }$ per pixel. Only a slide-level logistic-regression classifier was fitted to downstream labels, using raw embeddings, balanced class weights, L2 regularization (C = 1), the lbfgs solver and a maximum of 10,000 iterations, following the released few-shot probe configuration. Results are summarized in Supplementary Table S6.

## Benchmark construction and evaluation

Question generation proceeds from a case summary to individual viewing segments. Gemini-3-Flash first receives a $\hphantom { 0 } 3 \times 3$ montage, the session transcript and available case context to summarize the findings. It then receives individual segmen images, attention overlays, aligned narration and the case summary to produce question–answer candidates. Questions and their accepted answers were manually reviewed against the source images, aligned narration and recorded viewing context. Items are scored by whether the selected option belongs to the accepted-answer set. Some items have multiple accepted options or a different number of options. The uniform random single-choice reference is therefore $| { \cal A } _ { i } | / | { \cal O } _ { i } |$ for item i, where $A _ { i }$ and $O _ { i }$ denote its accepted answers and available options; 25% applies only to four-option items with one accepted answer.

The four categories address evidence integration across fields or slides, rejection of an inapplicable premise, recognition of the tissue or viewing context, and selection of the next region for a diagnostic goal. Narration and viewer state provide the evidence used to construct the intended answers. Models were evaluated against a shared question bank and accepted-answer keys. The retained bank contains 371 questions: 128 on inapplicable premises, 62 on multi-slide reasoning, 59 on goal-directed navigation and 122 on environmental sensing (63 viewing-scale and 59 tissue-location questions; Supplementary Table S7). Source-item examples are shown in Supplementary Figure S5.

The question-source comparison includes an image-input comparison for Claude and GPT on PathMMU<sup>44</sup> and Path-Cowork, and a separate comparison of Haiku 4.5, Sonnet 5 and Opus 4.8 across PathMMU, textbook-derived and PathCowork skin questions. Question-only inputs contain the question and options; image-supported inputs additionally contain the relevant image or montage. The textbook condition used locally assembled diagnosis questions from Dermatopathology: Diagnosis by First Impression, fourth edition<sup>45</sup>. The three-model source comparison uses a sepa rate 100-item skin set per source. The four-category comparison includes Opus 4.8, Haiku 4.5, GPT-5.5, GPT-5.6-sol, GPT-5.4-mini and Gemini 3.1, together with one pathology resident. Accuracy is the proportion of evaluated questions answered correctly within each category. The archived multi-model scoring routine counts an attempted question without a parseable answer as incorrect, while excluding questions with no recorded attempt; evaluable denominators can therefore differ across models.

## Statistical analysis

Behavioral analyses summarize sessions, utterances and viewing segments at the level indicated in each panel. MIL performance is assessed with quadratic-weighted Cohen’s κ on six diagnostic categories. Learning-curve bands and fixed-size bars use the standard deviation across 20 seeds, calculated with divisor 20, divided by $\sqrt { 2 0 }$ . The auxiliary-weight sweep instead summarizes ten seeds. Stars in Figure ${ \pmb 4 } { \bf b } , { \pmb c } , { \pmb d }$ compare CoworkMIL with ABMIL using paired, two-sided Wilcoxon signed-rank tests on matched seeds, without adjustment for multiple comparisons. All 15 comparisons, including their effect directions and exact P values, are listed in Supplementary Table S5. Tests use the archived seed-level κ values, stored to four decimal places, and quantify variation across training runs on the fixed test set.

Agent scores and cross-task summaries are descriptive, with each task contributing its specified clinical endpoint. The Opus 4.8 image-input comparison on PathCowork uses 148 matched questions with archived answers in both input conditions. These questions were selected from the recorded-review bank by retaining items with answers in all four archived Opus/Haiku image-supported and question-only conditions. The other displayed image-input comparisons retain their source-level scores. Question-level scores can include multiple items from the same recorded case.

Author contributions. S. Wang and Z.H. conceived and designed the study. S. Wang, J.Y., S.L. and J.X. developed the software. M.V.R., S.K., E.B., M.N., W.D., C.M., T.C., A.W., Z.Y., D.A., S. Walle and X.X. contributed recorded slide reviews, including eye-tracking data and verbal descriptions. E.B. also assisted with early software debugging and refinement of the data collection protocol. O.P., E.B.L., Z.A., A.S., R.T., N.D., D.E. and G.J.N. provided advice on pathology domain knowledge. S. Wang and Z.H. wrote the manuscript. Z.H. supervised the study.

Acknowledgements. This study was approved by the institutional review board (IRB #857258) of the Perelman Schoo of Medicine at the University of Pennsylvania. We thank Jonathan Alex Miles, Annie Jin, Ikuko Hirai and Meaghan Dougher for contributing slide-review recordings to our previous Pathology-CoT study<sup>20</sup>, which were reused here. We acknowledge Karin Lindman, Jerónimo F. Rose, Martin Lindvall and Caroline Bivik Stadler for the DROID-skin dataset<sup>26,27</sup>, and Gordan Maras, Martin Lindvall and Claes Lundström for the LNCO2 dataset<sup>28</sup>, provided through the AIDA Data Hub.

The DROID-skin and LNCO2 datasets are copyright 2019 and 2020, respectively, Linköping University, Claes Lundström. We thank the Region Gävleborg Clinical Pathology and Cytology department for producing the LNCO2 collection. We acknowledge the Ivy Glioblastoma Atlas Project (Ivy GAP)<sup>32,33</sup>, developed through a partnership of the Allen Institute for Brain Science, the Ben and Catherine Ivy Foundation, and the Ben and Catherine Ivy Center for Advanced Brain Tumor Treatment. The atlas was supported by the Ben and Catherine Ivy Foundation; our analyses and conclusions do not necessarily reflect the Foundation’s views. Ivy GAP data are copyright 2015 Allen Institute for Brain Science and are available at https://glioblastoma.alleninstitute.org/. The results published here are in part based upon data generated by the TCGA Research Network: https://www.cancer.gov/tcga. We thank the specimen donors and contributing research groups.

Data availability. Example slide-review records, including segment metadata and images, are available in the PathCowork history directory. The full training corpus and its complete concept inventory are not included in the model release.

Code availability. PathCowork processing and review code is available on GitHub. Links to model weights are also provided at https://github.com/zhihuanglab/PathCowork.

Competing interests. The authors declare no competing interests.

## Video legends

## Main Video 1 | Interactive whole-slide review with PathCowork.

Screen recording of PathCowork examining a skin whole-slide image. After the user selects a slide and starts review, the agent surveys the tissue, localizes junctional features, inspects the dermis and candidate mitotic figures, and records findings with localized visual evidence before returning control to the user. The full 127.3-second recording is shown at its original speed, with cleaned audio of the original agent narration, English subtitles, and opening and closing title cards.

## Main Video 2 | Capturing pathologist behavior during skin-biopsy review.

Real-time screen recording of a pathologist examining a skin whole-slide image in TissueLab during a PathCowork data-acquisition session. The yellow ring displays gaze position as the reader pans and changes magnification. The original recorded discussion describes peripheral palisading, myxoid stroma and an infiltrative tumor edge, illustrating how spoken observations accompany the tissue being examined. The recording shows human review during data collection. English subtitles accompany the original audio. The full 48.1-s review is retained at its original speed; opening and closing title cards are added, and the audio level is increased for audibility.

## References

1. U.S. Food and Drug Administration. Radiological computer-assisted detection and diagnosis software. 21 CFR 892.2090, URL https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-892/subpart-B/section-892. 2090 (2025). Section (b)(2)(ii)–(iii).

2. Ding, T. et al. A multimodal whole-slide foundation model for pathology. Nat. Medicine 31, 3749–3761, DOI: 10.1038/s41591-025-03982-3 (2025).

3. Shaikovski, G. et al. PRISM: A multi-modal generative foundation model for slide-level histopathology. arXiv preprint arXiv:2405.10254 DOI: 10.48550/arXiv.2405.10254 (2024).

4. Crowley, R. S., Naus, G. J., Stewart, J., III & Friedman, C. P. Development of visual diagnostic expertise in pathology—an information-processing study. J. Am. Med. Informatics Assoc. 10, 39–51, DOI: 10.1197/jamia.M1123 (2003).

5. Drew, T. et al. More scanning, but not zooming, is associated with diagnostic accuracy in evaluating digital breast pathology slides. J. Vis. 21, 7, DOI: 10.1167/jov.21.11.7 (2021).

6. Ilse, M., Tomczak, J. & Welling, M. Attention-based deep multiple instance learning. In Proceedings of the 35th International Conference on Machine Learning (ICML), vol. 80, 2127–2136, URL https://proceedings.mlr.press/v80 ilse18a.html (PMLR, 2018).

7. Campanella, G. et al. Clinical-grade computational pathology using weakly supervised deep learning on whole slide images. Nat. Medicine 25, 1301–1309, DOI: 10.1038/s41591-019-0508-1 (2019).

8. Lu, M. Y. et al. Data-efficient and weakly supervised computational pathology on whole-slide images. Nat. Biomed. Eng. 5, 555–570, DOI: 10.1038/s41551-020-00682-w (2021).

9. Chen, R. J. et al. Towards a general-purpose foundation model for computational pathology. Nat. Medicine 30, 850–862, DOI: 10.1038/s41591-024-02857-3 (2024).

10. Vorontsov, E. et al. A foundation model for clinical-grade computational pathology and rare cancers detection. Nat. Medicine 30, 2924–2935, DOI: 10.1038/s41591-024-03141-0 (2024).

11. Huang, Z., Bianchi, F., Yuksekgonul, M., Montine, T. J. & Zou, J. A visual–language foundation model for pathology image analysis using medical Twitter. Nat. Medicine 29, 2307–2316, DOI: 10.1038/s41591-023-02504-3 (2023).

12. Lu, M. Y. et al. A visual-language foundation model for computational pathology. Nat. Medicine 30, 863–874, DOI: 10.1038/s41591-024-02856-4 (2024).

13. Xiang, J. et al. A vision–language foundation model for precision oncology. Nature 638, 769–778, DOI: 10.1038 s41586-024-08378-w (2025).

14. Lu, M. Y. et al. A multimodal generative AI copilot for human pathology. Nature 634, 466–473, DOI: 10.1038 s41586-024-07618-3 (2024).

15. Chen, Y. et al. SlideChat: A large vision-language assistant for whole-slide pathology image understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 5134–5143, URL https://openaccess.thecvf.com/content/CVPR2025/html/Chen\_SlideChat\_A\_Large\_Vision-Language\_ Assistant\_for\_Whole-Slide\_Pathology\_Image\_Understanding\_CVPR\_2025\_paper.html (2025).

16. Dong, N. et al. Reinforced auto-zoom net: Towards accurate and fast breast cancer segmentation in whole-slide images. In Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support, vol. 11045 of Lecture Notes in Computer Science, 317–325, DOI: 10.1007/978-3-030-00889-5\_36 (Springer, 2018).

17. Xiong, C., Chen, H., Sung, J. J. Y. & King, I. Diagnose like a pathologist: Transformer-enabled hierarchical attention-guided multiple instance learning for whole slide image classification. In Proceedings of the Thirty-Second International Joint Conference on Artificial Intelligence, 1587–1595, DOI: 10.24963/ijcai.2023/176 (2023).

18. Buckley, T. A. et al. Navigating gigapixel pathology images with large multimodal models. arXiv preprint arXiv:2511.19652 DOI: 10.48550/arXiv.2511.19652 (2025).

19. Xu, Z. et al. MMNavAgent: Multi-magnification WSI navigation agent for clinically consistent whole-slide analysis. arXiv preprint arXiv:2603.02079 DOI: 10.48550/arXiv.2603.02079 (2026).

20. Wang, S. et al. Pathology-CoT: learning visual chain-of-thought agents from expert whole-slide image diagnosis behaviour. Nat. Biomed. Eng. DOI: 10.1038/s41551-026-01739-y (2026).

21. Li, S. et al. A co-evolving agentic AI system for medical imaging analysis. arXiv preprint arXiv:2509.20279 URL https://arxiv.org/abs/2509.20279 (2025).

22. Moonshot AI. Kimi K2.6, URL https://huggingface.co/moonshotai/Kimi-K2.6 (2026). Official model card.

23. Yang, A. et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388 DOI: 10.48550/arXiv.2505.09388 (2025).

24. Neidlinger, P. et al. A deep learning framework for efficient pathology image analysis. Nat. Commun. 17, 5740, DOI: 10.1038/s41467-026-74918-9 (2026).

25. The Cancer Genome Atlas Research Network et al. The Cancer Genome Atlas pan-cancer analysis project. Nat. Genet. 45, 1113–1120, DOI: 10.1038/ng.2764 (2013).

26. Lindman, K., Rose, J. F., Lindvall, M. & Stadler, C. B. Skin data from the visual sweden project DROID. AIDA Data Hub, DOI: 10.23698/aida/drsk (2019).

27. Stadler, C. B. et al. Proactive construction of an annotated imaging database for artificial intelligence training. J. Digit. Imaging 34, 105–115, DOI: 10.1007/s10278-020-00384-4 (2021).

28. Maras, G., Lindvall, M. & Lundstrom, C. Regional lymph node metastasis in colon adenocarcinoma, second collection series. AIDA Data Hub, DOI: 10.23698/aida/lnco2 (2020).

29. Radford, A. et al. Robust speech recognition via large-scale weak supervision. In Proceedings ofthe 40th International Conference on Machine Learning, vol. 202, 28492–28518, URL https://proceedings.mlr.press/v202/radford23a. html (PMLR, 2023).

30. Salvucci, D. D. & Goldberg, J. H. Identifying fixations and saccades in eye-tracking protocols. In Proceedings of the 2000 Symposium on Eye Tracking Research and Applications, 71–78, DOI: 10.1145/355017.355028 (ACM, 2000).

31. Hu, E. J. et al. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, URL https://openreview.net/forum?id=nZeVKeeFYf9 (2022).

32. Allen Institute for Brain Science. Ivy glioblastoma atlas project. Dataset, URL https://glioblastoma.alleninstitute.org (2015). Accessed 22 October 2026.

33. Puchalski, R. B. et al. An anatomic transcriptional atlas of human glioblastoma. Science 360, 660–663, DOI: 10.1126/science.aaf2666 (2018).

34. Shao, Z. et al. DeepSeekMath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 URL https://arxiv.org/abs/2402.03300 (2024).

35. Williams, R. J. Simple statistical gradient-following algorithms for connectionist reinforcement learning. Mach. Learn. 8, 229–256, DOI: 10.1007/BF00992696 (1992).

36. Modella AI. Meet PathChat 2. Official product description, URL https://www.modella.ai/pathchat. Accessed 22 September 2026.

37. Epstein, J. I. et al. The 2014 international society of urological pathology (ISUP) consensus conference on Gleason grading of prostatic carcinoma: Definition of grading patterns and proposal for a new grading system. The Am. J. Surg. Pathol. 40, 244–252, DOI: 10.1097/PAS.0000000000000530 (2016).

38. Nechaev, D., Pchelnikov, A. & Ivanova, E. HISTAI: An open-source, large-scale whole slide image dataset for computational pathology. arXiv preprint arXiv:2505.12120 DOI: 10.48550/arXiv.2505.12120 (2025).

39. Bulten, W. et al. Artificial intelligence for diagnosis and Gleason grading of prostate cancer: the PANDA challenge. Nat. Medicine 28, 154–163, DOI: 10.1038/s41591-021-01620-2 (2022).

40. MahmoodLab. UNI2-h. https://huggingface.co/MahmoodLab/UNI2-h (2025). Model card; accessed 9 September 2026.

41. Zimmermann, E. et al. Virchow2: Scaling self-supervised mixed magnification models in pathology. arXiv preprint arXiv:2408.00738 DOI: 10.48550/arXiv.2408.00738 (2024).

42. Wang, X. et al. Transformer-based unsupervised contrastive learning for histopathological image classification. Med. Image Analysis 81, 102559, DOI: 10.1016/j.media.2022.102559 (2022).

43. Wang, X. et al. A pathology foundation model for cancer diagnosis and prognosis prediction. Nature 634, 970–978, DOI: 10.1038/s41586-024-07894-z (2024).

44. Sun, Y. et al. PathMMU: A massive multimodal expert-level benchmark for understanding and reasoning in pathology. In Computer Vision – ECCV 2024, Lecture Notes in Computer Science, vol. 15120, 56–73, DOI: 10.1007/978-3-031-73033-7\_4 (Springer, 2024).

45. Ko, C. J. & Barr, R. J. Dermatopathology: Diagnosis by First Impression (Wiley-Blackwell, 2022), 4 edn.

46. Ross, S., Gordon, G. & Bagnell, D. A reduction of imitation learning and structured prediction to no-regret online learning. In Proceedings of the Fourteenth International Conference on Artificial Intelligence and Statistics, vol. 15 of Proceedings of Machine Learning Research, 627–635, URL https://proceedings.mlr.press/v15/ross11a.html (2011).

47. Almeida, D. Introducing System One models & Jev. TypeSafe AI, URL https://typesafe.ai/blog introducing-system-one-models-and-jev (2026).

49. Bándi, P. et al. From detection of individual metastases to classification of lymph node status at the patient level: The CAMELYON17 challenge. IEEE Transactions on Med. Imaging 38, 550–560, DOI: 10.1109/TMI.2018.2867350 (2019).

48. Thinking Machines Lab. Announcing Tinker, URL https://thinkingmachines.ai/news/announcing-tinker/ (2025).

## Supplementary Information

## Supplementary Note 1 | Additional skin grounding and an ancillary adaptation study

The main grounding comparisons use supervised fine-tuning of Kimi K2.6 and Qwen3.6-27B on examples derived from the recordings. Each example pairs an image and target concept with a structured spatial response. The dermatopathology grounding evaluation in Figure 2e,f uses the adapted Kimi K2.6 localizer. Additional qualitative skin comparisons from the Qwen SFT development export are shown in Supplementary Figures S1 and S2; these examples retain their source mode configuration and displayed metrics. Supplementary Table S3 lists the backbones and configurations used in the additional experiments on data quantity, model size, numerical precision and training duration.

The main test set contains 50 image–concept pairs for each evaluated concept, including 350 pairs across the seven dermatopathology concepts. Training and test patients were separated before generating the examples. Both sets were constructed from behavior-derived candidates with manual spot-checking, augmentation and automatically generated negative examples. Recall uses reference-positive examples as its denominator, precision uses predicted-positive examples, and localization retains missed targets and unparseable outputs with a score of zero.

Claude Opus 4.7

Gemini 3.1

![](images/addbd69a3cfe0809f880ba6d9a89bb058f88bf763f4e8b3c36c4057c51cd44a0.jpg)

Supplementary Figure S1. Additional skin grounding examples for confluent growth and pagetoid spread. Rows show two examples of confluent melanocytic proliferation and one of pagetoid spread. Columns compare the reference annotation with a PathCowork SFT localizer and three frontier models. Reference regions are green and model predictions are red. IoU and Dice are percentages; “Not detected” denotes a negative presence prediction.

Pigment incontinence  |  nvm\_311  
![](images/b5585d3c770ede0bdc1f05a37677f9fbace16546f5564ca7e325f470639e84ed.jpg)  
Supplementary Figure S2. Additional skin grounding examples for pigment, junctional anatomy and nests. Rows show pigment incontinence, the dermoepidermal junction and a junctional melanocytic nest. Columns, colors and percentage metrics follow Supplementary Figure S1.

## Ancillary development methods

A separate small-model development study used Qwen3. $. 5 \cdot 4 \mathsf { B } ^ { 2 3 }$ with rank-8 low-rank adaptation<sup>31</sup> and thinking mode disabled. Its case-level split contained 19 training, three validation and three test cases, corresponding to 936, 148 and 148 message rows. Supervised training used Adam, learning rate $5 \times 1 0 ^ { - 5 }$ , batch size 2, maximum sequence length 8,192 tokens and 600 optimization steps. Validation was performed every 50 steps, and the step-450 checkpoint was selected. Selection considered polygon Dice, point-in-target rate, centroid distance, bounding-box intersection over union (IoU) and valid-JSON rate.

This development study also explored an appearance-aware reinforcement-learning objective after supervised training. Recorded attention can identify one instance among several plausible occurrences of a finding in the same field. The reward combined center hit (weight 0.25), normalized centroid proximity (0.20), appearance similarity (0.20), bounding-box IoU (0.15), polygon Dice (0.10) and area agreement (0.10). Appearance similarity was estimated from CIELAB channel statistics and histograms. Training used 25 policy-gradient steps, learning rate $1 0 ^ { - 5 }$ , four prompts and four sampled responses per prompt, temperature 0.2 and maximum response length 256 tokens. This ancillary SFT-plus-RL experiment is separate from the supervised-only comparisons in the main figures.

## Supplementary Note 2 | Detailed agent outcomes across five organs

Supplementary Figure S3 reports the organ-specific endpoints underlying the overview in Figure 3e. The evaluations comprise skin biopsies for melanoma versus dysplastic nevus (20 cases, 69 WSIs), prostate needle biopsies for cancer detection and ISUP grading (20 slides), breast resections for Nottingham grading (39 matched cases), brain resections for glioblastoma versus lower-grade glioma (30 cases, 15 per class) and colon-resection lymph nodes for metastasis detection (77 slides, 10 positive and 67 negative).

Brain cases were drawn from TCGA-GBM and TCGA-LGG<sup>25</sup>, breast and melanocytic cases from HISTAI<sup>38</sup>, prostate slides from PANDA<sup>39</sup>, and colorectal lymph-node slides from LNCO2<sup>28</sup> (Supplementary Table S4). Evaluation cases were unseen during grounding training and skill construction. PANDA ISUP labels supplied the prostate grading reference, and LNCO2 annotations supplied the metastatic-node references.

The organ-specific experiments were conducted at different times using the strongest models then available in each agent framework. The prostate comparators used Claude Code with Sonnet and Codex with GPT-5.5; their evaluation runner allowed up to ten image crops and a 480-s wall-clock limit per slide. Codex used GPT-5.6-sol for brain and colon, with high reasoning effort for colon. The breast Claude Code comparator used Sonnet 5, and the brain PathCowork run used v4.4 High. PathCowork completed the prescribed skill steps without a task-level timeout, as described in Methods.

In skin, PathCowork achieved melanoma recall of 100% and specificity of 60%, compared with 40% recall and 90% specificity for both general-purpose agents. Its higher overall accuracy therefore accompanied lower specificity. In prostate, quadratic-weighted ISUP κ was 0.38, 0.06 and 0.30 for PathCowork, Claude Code and Codex, respectively; the corresponding grade mean absolute errors were 1.40, 1.74 and 2.15. Prostate detection uses all 20 slides, with the Claude Code timeout counted as a failure (17/20, 85%); its grading κ and MAE retain the 19 valid predictions.

The brain comparison uses 30 cases (15 GBM and 15 LGG) for every agent. PathCowork correctly classified 13/15 GBM and 13/15 LGG cases (26/30 overall); Claude Code classified 3/15 and 15/15 correctly (18/30); Codex classified 6/15 and 15/15 correctly (21/30). Missing reports numbered zero, two and zero, respectively, and remain in the denominator as incorrect outcomes. Mean times were 20, 332 and 285 seconds per slide, respectively.

The breast comparison uses one matched set of 39 cases for all agents. In the order PathCowork, Claude Code and Codex, overall grade accuracy was 28/39 (71.8%), 17/39 (43.6%) and 21/39 (53.8%). Correct component scores were 26, 21 and 22 for tubule formation; 26, 22 and 24 for nuclear pleomorphism; and 19, 14 and 20 for mitotic count, all out of 39. Grade-specific recall was 5/10, 1/10 and 6/10 for grade 1; 10/14, 9/14 and 9/14 for grade 2; and 13/15, 7/15 and 6/15 for grade 3. PathCowork’s largest grade-specific advantage was therefore in grade 3, while Codex had higher grade 1 recall and correctly scored one more case for mitotic count. Per-slide costs were \$0.15, \$1.98 and \$1.25, and times were 106, 300 and 192 s, respectively (Supplementary Figure S3).

For LNCO2, Claude Code achieved 87.3% node detection recall, called 12/15 ground-truth tumour nodes positive (80% end-toend per-node metastasis recall), and called 101/101 non-tumour nodes negative (100% per-node specificity). Missed tumour nodes remain in the 15-node denominator as false negatives. Missing reports numbered zero, six and zero for PathCowork, Claude Code and Codex, respectively. Claude Code’s slide-level recall was 6/10 (60%) and specificity was 62/67 (92.5%); its two missing positive-slide reports and four missing negative-slide reports counted as failures. Codex achieved slide-level recall of 9/10 (90%) and specificity of 67/67 (100%).

Costs are reported per slide at the recorded API prices; the Codex colon cost was \$1.04 per slide. Times are wall-clock values per slide, summarized by the median for colon and the mean for other tasks.

![](images/0c6801a4ec89dcfb3fd6f7bbd260a02ded1209545e53ed2d7f631afedd3ced1f.jpg)

Supplementary Figure S3. Detailed agent outcomes across five organs. Rows compare PathCowork (purple), Claude Code (orange) and Codex (grey) on primary and secondary endpoints; shaded panels identify the primary endpoints used in Figure 3e. Colon node detection recall measures the proportion of reference lymph nodes located. In the breast row, Tub., Nuc. and Mit. denote tubule formation, nuclear pleomorphism and mitotic count; G1–G3 denote reference grades. Costs and wall-clock times are per slide; time is the median for colon and the mean for other tasks. Arrows indicate the direction of improvement. Cohorts, model configurations and scoring rules are detailed in Supplementary Note 2.

## Supplementary Note 3 | General navigation learned from pathologist behavior

The five agents in Figure 3 turn recorded case experience into procedures for recurring tasks. Open-ended questions can require findings or relationships beyond this repertoire, including examinations only partly represented in the available recordings.

PathCowork-general learns reusable viewing actions—surveying tissue, moving between fields, changing magnification and returning to a wider context—to gather evidence beyond a predefined skill. The proposed workflow assigns navigation to this learned policy and interpretation to an LLM, extending the use of recorded behavior to questions for which a complete procedure has yet to be compiled. The experiments below evaluate navigation and evidence collection; combined question answering is a future application.

Two complementary uses of recorded behavior. Supplementary Video 3 shows a recorded example of the learned navigator examining a lymph-node slide. We refer to the task-specific approach in Figure 3 as PathCowork-skill and the learned navigation approach as PathCowork-general (Supplementary Figure S4a). The former encodes examination steps explicitly; the latter learns successive viewing decisions from behavioral records and reference trajectories. Both reuse pathologist experience. A stepwise agent such as Claude Code or Codex instead invokes the LLM to choose successive viewing actions. Here, “general” denotes a reusable navigation policy shared across tasks.

Indicative navigation time and cost. The indicative values in Supplementary Figure S4a cover navigation decisions. “Instant” and “Free” denote script overhead. The learned-policy estimate of 300 ms is based on a 318-ms forward pass of an earlier 384-pixel, 20-slot prototype on an M5 Pro with MLX bf16. Its hosted-inference equivalent is approximately US\$0.00025 per decision: 753 image-plus-prompt tokens at the Tinker 4B prefill rate of US\$0.33 per million tokens, accessed 21 September 2026. LLM estimates are 20–30 s and approximately US\$0.03 per step, based on US\$1 per 30 steps. These estimates cover decision computation, excluding image loading, downstream interpretation, training and hardware costs.

From recorded examination to navigation targets. The general navigator combines processed pathologist gaze and viewport records with trajectories generated by a reference policy on annotated slides. Both sources are mapped to the same viewing scales and action vocabulary. Human records preserve the operations taken during examination; reference trajectories supply broader image coverage and examples of surveying tissue, approaching an annotated focus and returning to continue the survey. Perturbed reference trajectories provide corrective targets in states outside the usual path, motivated by the distribution-shift problem addressed by dataset aggregation for imitation learning<sup>46</sup>. Images and spatial targets are transformed together under rotations and reflections. Before subsampling for training, the mixed-supervision pool contained 40,000 records, including 2,021 from human recordings.

An exploratory comparison of behavioral supervision. An earlier model comparison added 1,916 human navigation records to the same 5,000 reference-navigation and 6,000 perception records used by a control. Both models used one training epoch and the same training recipe, although adding records also increased the number of optimization steps. On 286 held-out views with a defined human action, agreement with that action increased from 24.8% to 60.1%. Among the 181 views followed by a human pan, the control predicted pan zero times and the model with human records predicted it 105 times. Human records increased use of the pan operation, although agreement remained below the 63.3% majority-action baseline and repetitive panning persisted. Each condition used one run of an earlier model and state representation.

A local model for successive viewing decisions. The general navigator uses a four-level ladder: the tissue overview followed by square fields of 4.0, 1.33 and 0.44 mm; the survey field is 2.5 mm for lymph-node slides. Each field is rendered at 384 × 384 pixels and divided into a 4 × 4 grid. The state records viewing scale, remaining steps, tissue and coverage fractions, areas already examined at a deeper level, directions containing unsurveyed tissue and the last three actions. Per-cell survey coverage is computed before marking the current arrival, allowing a new field to be distinguished from a revisit.

The navigator selects among zooming into a grid cell, zooming out, panning in one of eight directions and stopping. Its general-model template contains 29 answer slots: 16 cell-level attention outputs, three action outputs, nine field-description outputs and one unused deferral output. A single forward pass supplies logits at fixed answer positions; softmax over the allowed answer letters gives each slot’s distribution, without autoregressive text generation. This use of predefined decision outputs follows the System One design principle described for Jev<sup>47</sup>; our implementation adapts a vision–language model to pathology images and viewing behavior. It uses Qwen3.5-4B<sup>23</sup> with rank-32 LoRA<sup>31</sup>. The evaluated general model was trained through Tinker<sup>48</sup> for one epoch on 16,000 navigation and 12,000 perception records, with batch size 32 and peak learning rate $2 \times 1 0 ^ { - 4 }$ Cross-entropy is restricted to defined answer positions; sampling an answer from a soft target distribution matches that target in expectation. The best validation checkpoint was the end-of-epoch checkpoint. Local inference used merged bf16 weights with MLX.

Coverage bookkeeping and evidence selection. Deterministic rules constrain the model’s choices: they restrict repeated entry into already examined cells, veto reverse pans when another direction contains unsurveyed tissue, and return to a wider view when continued panning adds no coverage. At the end of a review, distinct non-overview views are grouped by tissue fragment, and up to six per fragment are retained according to the maximum of their predicted tumor, interface and necrosis scores. Survey views remain eligible to preserve lower-power architecture. The retained images form the evidence set for subsequent interpretation.

Evaluation on held-out lymph-node units. The general-model evaluation used 12 CAMELYON17 lymph-node units<sup>49</sup>: four labelled macrometastasis, four micrometastasis and four negative. Patients were held out from the CAMELYON17 training sources; training images included other patients from the same dataset. On 400 navigation records, the 6,400 cell outputs achieved AUROC 0.965 and average precision 0.455, compared with 0.886 and 0.101 when the image was replaced by a white field while retaining the state. Operation agreement was 88.2% versus 86.8% in that control; pan-direction agreement was 41.7% versus 35.4% over 223 pan-labelled records. The image contribution varied by output and scale: for overview-level cell outputs, average precision was 0.108 with the image and 0.186 without it.

What the resulting reviews captured and missed. Across one review per unit, median tissue survey coverage was 91%, and seven of 12 reviews stopped within the nominal 60-step budget (Supplementary Figure S4c). Four of eight metastatic units received both a mid-power visit overlapping an annotated focus and a high-power read on the lesion. The selected evidence set contained a view overlapping the annotation in all eight, including survey views from the four units without a high-power lesion read. Representative successes and failures are shown in Supplementary Figure S4b. Eight of 12 reviews had no immediate reverse pan; five reached the budget, and two had no mid-power visit to any qualifying tissue fragment. Recorded local review times ranged from 8.17 to 35.17 s, excluding downstream LLM interpretation. These exploratory results identify coverage, revisiting and stopping as unresolved behavioral problems. The navigator weights are released a https://github.com/zhihuanglab/PathCowork; its training corpus is not included in the release.

Trajectory: dark → yellow = earlier → later; ○ start, ■ end.

Scale bar, 5 mm

## Navigationa

PathCowork-skill

![](images/bf42db209301a59e28b22804c21120058fb63e73bffadd412fa3b8ed136d2bb9.jpg)  
Follow rules

PathCowork-general

![](images/2bbdcc6e10b6a21c1de744c8911576d7c9a7899d045cec01feda1f3febaa34b0.jpg)  
Direct action

Claude Code / Codex

![](images/ae78dba61103f118715d12c5b960966f0e2406dbd2393b42811a36b3f1ac620d.jpg)  
Reason then act

Time / step

Instant

≈300 ms

≈20–30 s

Cost / step

Free

≈\$0.00025

≈\$0.03

Generalizability

Task-specific

General

General

## b Representative navigation trajectories

P073 / 1Macrometastasis

![](images/253fb1d6a6a706e8063b3c19c9da58efe9bf0829870193aa389cd8ad88392dfa.jpg)

P042 / 3 Macrometastasis

![](images/3c1ac41176fe304d688942d460677539232af85df83d59901d6853811ae17f40.jpg)

P010 / 4 Micrometastasis

![](images/4333df2126322931233da5412cab94f76770888f5bc95b8c50079c82d4f27480.jpg)  
P048 / 1 Micrometastasis

![](images/ecb054331dbd3c9b840e57a12d5d10d1f6049b852800e81f0dcd2583c0aabf49.jpg)  
P055 / 0 Negative

![](images/25b7f9cb18773d507326dc7e44b7689770640459f3162e386c6877aa29a9eb49.jpg)  
P070 / 2 Negative

![](images/97c4521ba80b2e9ee5744d4256f9de05ff95ddfe64c5d724c8d752b76495aeae.jpg)

## c Navigation and selected evidence across 12 held-out units

<table><tr><td>Unit</td><td>Category</td><td>Lesion mm²</td><td>Entries</td><td>Stopped</td><td>Survey %</td><td>HP reads on / all</td><td>Pieces visited</td><td>Foci touched</td><td>Evidence on / all</td><td>Reverse pans</td><td>Revisits</td><td>Time s</td></tr><tr><td>P015/2</td><td>macro</td><td>1.011</td><td>25</td><td>Yes</td><td>92</td><td>0/1</td><td>2/3</td><td>0/1</td><td>2/14</td><td>0</td><td>0</td><td>13.40</td></tr><tr><td>P042/3</td><td>macro</td><td>3.036</td><td>32</td><td>Yes</td><td>91</td><td>3/3</td><td>2/3</td><td>3/3</td><td>12/14</td><td>2</td><td>5</td><td>16.42</td></tr><tr><td>P073/1</td><td>macro</td><td>45.368</td><td>61</td><td>No</td><td>47</td><td>2/3</td><td>2/4</td><td>2/4</td><td>12/12</td><td>0</td><td>3</td><td>35.17</td></tr><tr><td>P099/4</td><td>macro</td><td>8.310</td><td>61</td><td>No</td><td>81</td><td>1/1</td><td>1/2</td><td>1/1</td><td>6/12</td><td>0</td><td>10</td><td>31.36</td></tr><tr><td>P010/4</td><td>micro</td><td>0.814</td><td>51</td><td>Yes</td><td>91</td><td>1/3</td><td>5/5</td><td>2/3</td><td>7/24</td><td>0</td><td>4</td><td>28.05</td></tr><tr><td>P021/3</td><td>micro</td><td>0.054</td><td>29</td><td>Yes</td><td>91</td><td>0/1</td><td>1/2</td><td>0/2</td><td>1/12</td><td>0</td><td>0</td><td>16.20</td></tr><tr><td>P048/1</td><td>micro</td><td>0.074</td><td>61</td><td>No</td><td>80</td><td>0/0</td><td>0/6</td><td>0/1</td><td>1/26</td><td>0</td><td>11</td><td>31.55</td></tr><tr><td>P089/3</td><td>micro</td><td>0.017</td><td>23</td><td>Yes</td><td>94</td><td>0/0</td><td>0/2</td><td>-</td><td>2/12</td><td>0</td><td>0</td><td>11.44</td></tr><tr><td>P000/4</td><td>negative</td><td>0.000</td><td>15</td><td>Yes</td><td>98</td><td>0/0</td><td>1/2</td><td>一</td><td>0/7</td><td>1</td><td>2</td><td>8.17</td></tr><tr><td>P025/1</td><td>negative</td><td>0.000</td><td>61</td><td>No</td><td>50</td><td>0/11</td><td>1/2</td><td>一</td><td>0/6</td><td>1</td><td>2</td><td>35.14</td></tr><tr><td>P055/0</td><td>negative</td><td>0.000</td><td>31</td><td>Yes</td><td>93</td><td>0/0</td><td>0/2</td><td>一</td><td>0/12</td><td>1</td><td>4</td><td>15.22</td></tr><tr><td>P070/2</td><td>negative</td><td>0.000</td><td>61</td><td>No</td><td>67</td><td>0/1</td><td>2/4</td><td>一</td><td>0/12</td><td>0</td><td>9</td><td>33.64</td></tr></table>

Supplementary Figure S4. General navigation learned from pathologist behavior. a, PathCowork-skill follows a compiled examination procedure; PathCowork-general predicts viewing actions from a learned policy; stepwise Claude Code or Codex agents generate an LLM response before acting. Icon sizes are schematic. Time and cost are indicative navigation estimates (Supplementary Note 3). b, Six patient-held-out CAMELYON17 units, showing tissue, annotated tumor, successive fields and high-power reads. Other patients from this dataset contributed training data. Scale bars, 5 mm. c, All 12 evaluated units, one review each with the supervised navigator, deterministic coverage rules and a nominal 60-step budget. Entries include the terminal observation; budget-limited runs contain 61. Stopped indicates an unforced stop within budget. Survey measures tissue coverage at survey level or higher. High-power (HP) reads count 0.44-mm fields on lesion over all HP fields. Pieces counts fragments of at least 2 mm<sup>2</sup> visited at 1.33 mm or higher; foci counts annotations of at least 0.02 mm<sup>2</sup> touched at these scales. A dash denotes no eligible focus. Evidence counts retained views overlapping tumor over all retained views. Reverse pans and repeated views are separate counts. Time excludes downstream LLM interpretation.

## Supplementary Note 4 | Frozen tile selection on the matched prostate cohort

We compared behavior-derived spatial supervision with EAGLE, an alternative that selects tissue using attention from a pretrained model<sup>24</sup>. The comparison used the fixed 30-slide PANDA test cohort from Figure 4, containing five slides in each ISUP grade group from 0 to 5. Both EAGLE resolutions used exactly the same 20 training subsets as the N = 20 CoworkMIL evaluation, sampled from the 65-slide training pool.

For each resolution, frozen CTransPath features<sup>42</sup> were ranked by frozen CHIEF attention<sup>43</sup>, and up to 25 selected tiles were re-encoded with Virchow2<sup>41</sup>. The mean of their class-token embeddings formed the slide representation. A logistic-regression classifier was fitted to the 20 training slides, with the encoders and tile selector frozen. EAGLE achieved mean quadratic-weighted κ of 0.430 at 0.5 µm per pixel and 0.435 at 2.0 µm per pixel, with mean absolute grade errors of 1.225 and 1.298, respectively. On these same training subsets and test slides, Virchow2-CoworkMIL achieved κ = 0.486 and mean absolute error 1.178, compared with 0.385 and 1.347 for Virchow2-ABMIL (Supplementary Table S6).

EAGLE uses 1,280-dimensional class-token features and mean pooling, whereas the Virchow2 MIL models use 2,560- dimensional concatenated class-token and mean patch-token features with a trained aggregator. Spatial sampling also differs, and CoworkMIL additionally receives grounding-derived supervision during training. Statistical summaries follow the fixed-cohort analysis described in Methods.

## Supplementary Note 5 | Benchmark sources and coverage of recorded clinical checks

## Supplementary Figure S5 compares illustrative questions from PathMMU<sup>44</sup>, a textbook atlas and recorded review.

Generated questions and accepted-answer keys underwent manual review using the source images and recorded context. The retained question-bank composition is given in Supplementary Table S7; environmental sensing combines viewing-scale and tissue-location questions. The separate source comparison uses 100 questions from each source, and the paired Opus image-input comparison uses 148 recorded-review questions, as described in Methods.

Requires locating the field within its specimen. Distractors are other sites on this same slide

![](images/574e443320a851ed994e99acb735ef5fe64c93b5fd0fa99c743dd7469c1abe2c.jpg)  
INTERNET PathMMU Based on the cellular morphology observed in the image, which feature does NOT suggest a benign melanocytic lesion?  
A) Clear cytoplasmic boundaries B) Even distribution of cells C) Variation in cell size and shape key D) Individual cells nested within the stroma  
Tests recognition of cellular atypia. Question-only accuracy (Opus 4.8): 59% on this set

![](images/a63e78c040e894435f7ec9be6423b117a4ad69657292cffcb046b0bfbcd748bd.jpg)  
TEXTBOOK In-house atlas What is the most likely diagnosis for this dermatopathology image?  
) Malignant melanoma ← key B) Deep penetrating nevus ← opus-5<sub>C)</sub> <sub>Blue</sub> <sub>nevus ←</sub> <sub>sonnet-5</sub> D) Mongolian spot Tests diagnosis among melanocytic mimics. Question-only accuracy (Opus 4.8): 28% on this set

## Questions from recorded slide review

![](images/0c14e497fcd822594e52064cd85623b76b56c4ceb844ef981ae502601b5eb2a9.jpg)  
<sup>A)</sup> <sup>2xB)</sup> <sup>4x</sup>C) 10x <sub>D)</sub> <sub>20x</sub> <sub>E)</sub> <sub>40x</sub> 10x ← key 20x ← key )40x  
No scale bar is given. Roughly what magnification are you viewing this field at?  
Requires estimation of the current viewing scale. Two adjacent bands accepted

![](images/e5c00572a139dee217c8547126ffb4d976595e8576a03ae006d8d96b0b894d6c.jpg)  
WHERE TO LOOK NEXT\_ACTION Given the need to evaluate the dermoepidermal junction for melanocytic nests, which region should you examine next?  
A) region A B) region B C) region C D) region D ← key Requires choosing a region for a stated diagnostic goal. Regions are labelled on the image itself

![](images/bdaa25e74419825d4cc1f89b835e70ea7e43b04c563c295b9d4a6de58b6b2fcd.jpg)

FALSE PREMISE HALLUCINATION   
Does the melanoma   
infiltrate the   
dermo-epidermal junction   
in this case?

![](images/5f4f4b8639e13c8eb4e3654789742a52d810154d0b96b39aec0afefd9b647fee.jpg)  
A) the dermal inflammatory infiltrate of the skin ← key B) the melanocytic lesion of the skin C) the dermis beneath the lesion of the skin D) the dermo-epidermal junction of the skin

## Where in the specimen is this field taken from?

## WHOLE CASE MULTI\_SECTION

![](images/575ce92a2a0a39ed19b15802640881894f839a8be55259165d217ea94e51d42d.jpg)  
Based on the atypical melanocytic proliferation at the dermal-epidermal junction with pagetoid spread, what is the most appropriate preliminary diagnosis?

C) Melanoma in situ ← key

A) Yes, the melanoma infiltrates the dermo-epidermal junction, demonstrating prominent pagetoid spread into the overlying epidermis.   
B) No, the melanoma does not infiltrate the dermo-epidermal junction and is entirely confined to the underlying dermal tissue.   
C) The question does not apply to this specimen — no such lesion is present here ← key   
D) Cannot be determined from this field — a deeper level or another block would be needed

## How to read a panel

Green, bold, ← key   
the accepted answer. Two options carry the key where the item accepts either.   
Orange, ← opus-5 / ← sonnet-5   
the option that model chose — every orange mark is a wrong answer. Italic line   
what the item is testing   
Grey line   
source-level results or details of the example.

Supplementary Figure S5. Examples of questions from three benchmark sources. Top, PathMMU and textbook items. Below, recorded-review items address viewing scale, specimen location, navigation, multiple sections and a false premise; the first two are environmental-sensing subtypes in Figure 5. Green marks accepted answers and orange marks the named models responses. The magnification example retains the source version’s five options and two accepted adjacent bands. Question-only percentages describe source-level Opus 4.8 results.

## Coverage of recorded clinical checks

We asked whether questions constructed from recorded review retained checking content that was less readily represented by a final impression or existing question banks. The reference contained 347 selected check occurrences from 103 reader–case reviews across 21 case groups; checks expressed by different readers were retained separately. With up to two questions per review, narration-derived questions covered 31.5% of the reference, compared with 9.8% for questions generated by the same model from the reader’s final impression. Retrospective selection of the best-matching pair yielded 14.8% coverage from the 210-item PathMMU dermatopathology bank and 0.0% from the corrected 160-item textbook-derived bank (Supplementary Figure S6). Narration-derived coverage was 16.7 percentage points higher than PathMMU (95% case-bootstrap interval, 8.5–24.8) and 31.5 points higher than the textbook bank (25.4–37.8).

Using all questions in each bank gave 14.8% coverage for PathMMU and 0.0% for the textbook bank. Restricting the latter to the corrected 100-item subset used in the main source comparison gave 0.0%. No textbook match passed both joint coding and verbatim-support validation. The evaluated bank contained diagnosis-only questions; identifying a diagnosis was not credited as testing every observation or examination action that could inform it. Separately, the two coders assigned textbook best-pair coverage of 1.3% and 14.4%, respectively; the primary analysis required their agreement on the same requirement–question match.

Recorded content included examination plans as well as morphological observations. A search for lymphovascular invasion and a proposed stain to exclude occult invasion were retained in narration-derived questions. Generation remained incomplete: a recorded pigment check was omitted by both generated question sets. The comparison measures retention of recorded checking content under a limited question budget.

The zero textbook estimate was sensitive to the quotation check: both coders matched a scar question, but one supplied a composite description of its options rather than a verbatim supporting phrase. In a post-hoc analysis retaining all in-scope model-proposed matches without requiring verbatim supporting phrases, coverage was 34.4% for recorded review, 11.3% for final impressions, 15.3% for PathMMU and 0.2% for the textbook bank. This sensitivity rule was applied to all conditions without new inference; the ordering was unchanged.

Coverage of recorded clinical checks

103 reader–case reviews · 21 case clusters · 347 check occurrences

![](images/692a75d6ba426113275b4b21080c1bd33a6afbaf9108a7399cf6bb97212451de.jpg)  
Up to 2 questions per review; equal case weighting.  
Primary: joint matches with verbatim support. Intervals: case bootstrap.  
\* Textbook: 0.2% without verbatim-support filtering (sensitivity analysis).

Supplementary Figure S6. Coverage of recorded clinical checks by four question sources. For each of 103 reader–case reviews, up to two questions were generated from the recorded narration or a final-impression summary, or retrospectively selected to maximize coverage from the archived PathMMU (210 items) or corrected textbook-derived (160 items) bank. The reference contains 347 selected check occurrences, including similar checks recorded by different readers, across 21 case groups. Coverage requires agreement of two model-based coders on the same check–question match, with verbatim supporting text. Dots show case means after averaging readers; filled diamonds show the equal-case mean and bars the 95% percentile interval from 10,000 case-bootstrap resamples. Hollow diamonds show full-bank coverage without the two-question limit. The textbook bank consists of atlas-derived diagnosis questions. Asterisk: textbook coverage is 0.2% in the post-hoc sensitivity analysis without verbatim-support filtering; the same sensitivity rule was applied to all conditions. The analysis measures textua correspondence between recorded checks and question content.

## Content-coverage methods

This content-coverage experiment was separate from the image-supported benchmark construction described in the main Methods. Five readers examined the same 30 whole-slide images, grouped under 21 case prefixes. Companion sections and repeat visits were pooled within each reader–case combination. Before question generation, a language-model assistant extracted explicit observations, checking actions and requests for additional information from the transcripts, retaining source quotations and timestamps. Selection across recognition, characterization, examination completion and additional-information categories retained at most four checks per combination. Repeated checks within a combination were merged, while similar content expressed by different readers remained separate. Two combinations without an explicit check were excluded, leaving 103 evaluable reviews and 347 check occurrences. The resulting reference was a capped set of assistant-extracted checks.

Claude Sonnet 5 generated up to two four-option questions from either the cleaned narration or a source-supported summary of the same reader’s final impression, using identical instructions and allowing abstention. The impression summaries were constructed from the transcripts for this experiment and retained uncertainty and associated findings. Generation used text-only inputs, and the resulting candidate questions were retained as generated.

The comparison banks contained 210 archived PathMMU dermatopathology questions<sup>44</sup> and 160 questions constructed from Dermatopathology: Diagnosis by First Impression<sup>45</sup>. The latter were locally assembled diagnosis questions. We used corrected intended answers linked to the printed atlas panel letters and preserved the original stems and choices. All 160 stems asked for the most likely diagnosis. The corrected 100-item subset used in the main source comparison was analyzed separately. Bank sizes and question formats therefore differed.

GPT-5.6-Sol and Claude Haiku 4.5 independently matched checks to candidate questions. Source labels and reader identities were withheld; items had opaque identifiers. Intended answers distinguished the central tested subject from distractor mentions The rubric required correspondence to both the clinical target and the checking operation. For example, identifying invasion did not automatically cover deciding to obtain an additional stain. Each proposed match included a literal phrase checked against the question text. Primary coverage required both coders to accept the same check–question match and provide valid verbatim support; individual-coder results were retained as sensitivity analyses. Generation and matching used subscription CLIs, without tools. The textbook extension reused the frozen rubric and settings in separate textbook-only packets. Original-condition matches were retained, so packet context and evaluation occasion differed between the original and added conditions.

For each review, exhaustive retrospective selection identified up to two bank questions maximizing the union of covered checks. Selection used the recorded checks as the retrieval reference. Full-bank unions summarized coverage using every question in each bank. Coverage proportions were averaged equally across readers within case, then equally across the 21 cases. Paired differences and mean intervals used 10,000 case-bootstrap resamples (seed 20260920). Narration supplied both the reference and the richer generation input, making within-record content retention the evaluation target.

## Supplementary Note 6 | Data scaling and concept transfer

We examined training-data dependence in a prostate low-power triage task using a Qwen3.5-27B localizer. The target vocabulary contained eight architectural patterns: small crowded glands, infiltrative growth, glands out of place, fused or poorly formed glands, cribriform architecture, glomeruloid architecture, sheets or single cells, and comedonecrosis. Training sets contained 10, 20, 30 or 51 annotated WSIs, with two epochs at each size.

Detection recall increased from approximately 61% at 10 WSIs to 90% at 51 WSIs, while localization IoU remained between approximately 41% and 46% (Supplementary Figure S7). All eight concepts were represented in the smallest set, and the number of training polygons increased from 752 to 4,024. Additional examples primarily improved detection coverage within this concept vocabulary.

![](images/403b919edee78c6bede620d4e1c7e084bd049ec879db8dcbe766dd8d34415648.jpg)  
Supplementary Figure S7. Data scaling of prostate concept grounding. Left, detection recall, localization intersection ove union (IoU) and output parse rate as the number of annotated training WSIs increases, with two training epochs at each size. Right, numbers of training polygons and represented concepts. The smallest set includes all eight concepts; the number of polygons increases from 752 to 4,024.

The per-concept plots show this distinction for six architectural patterns (Supplementary Figure S8). For example, recall for infiltrative growth increased from approximately 36% to 100%, and recall for fused or poorly formed glands from approximately 38% to 88%, across the plotted training sets. Changes in IoU were smaller and varied in direction across concepts.

![](images/a32784bac4ec9a8f84f92d7161a9869898b416f9e4ad29c1e2f0d08448ba84f2.jpg)

![](images/f922f06049549d9e2307ce90f2bf5a0258c4239fcee3930fb3750703ee7d66e4.jpg)  
Supplementary Figure S8. Detection and localization by concept and training-set size. Recall (left) and localization IoU (right) are plotted against the number of training regions for six prostate architectural patterns: small crowded glands, cribriform architecture, glands out of place, infiltrative growth, sheets, and fused or poorly formed glands. Colors identify concepts; dotted lines show the fitted trends against log-transformed region counts.

Training budget also affected the comparison. Under a fixed budget of 100 optimization steps, parse rate declined from approximately 97% to 69% as the dataset increased; the largest fixed-step run received about 0.28 epochs. Parse rate remained near 100% in the two-epoch series (Supplementary Figure S9). The fixed-step runs hold the optimization budget constant; the equal-epoch runs expose each training example a comparable number of times.

![](images/77f805fc9ba9b32be0c1d27c06396e111672e4d4405fdca4cd47910807b864df.jpg)

![](images/be73e5f3a0ce5055470f6c2c0dedd7d4f5781dfe7659f512b389cd0f289e96c6.jpg)  
Supplementary Figure S9. Fixed-step and equal-epoch training comparisons. Output parse rate (left) and localization IoU (right) for a fixed budget of 100 optimization steps (red) and two epochs at each dataset size (green). The largest fixed-step run receives approximately 0.28 epochs.

The adapted model achieved higher IoU than the displayed frontier baselines on each of the six plotted concepts (Supplementary Figure S10). Individual prostate fields show substantial variation, including non-overlapping predictions and unparseable responses (Supplementary Figures S11 and S12).

PathCowork vs frontier baselines — per-concept loU  
![](images/54b0e451c49e4d81c20588e1fb1e9db9af41f3dbd4e48ded16ff1e8309246d8d.jpg)  
Supplementary Figure S10. Localization of six prostate architectural concepts. IoU for the adapted PathCowork localizer and Gemini 3.1, GPT-5.4 and Opus 4.7. Concepts are small crowded glands, infiltrative growth, glands out of place, fused or poorly formed glands, cribriform architecture and sheets.

![](images/7915b1702cb5c0b86e6c4af459a53cae48b3e1fc5f22dabe3d67845d87604887.jpg)

Supplementary Figure S11. Qualitative grounding comparisons in prostate tissue. Two examples of small crowded glands and two of fused glands are shown with their reference regions and model outputs. Outlines show reference and predicted regions. Values below each field are example-level IoU percentages. FOV, field of view.

Fused glands  |  FOV 2.17 mm  
![](images/7d397645a929c042cf6ec22da61d199a8446a80891d981c63c0456d7c7a137a4.jpg)  
Supplementary Figure S12. Additional fused-gland grounding examples. Three further prostate fields are shown with the same column order as Supplementary Figure S11. Values are example-level IoU percentages. “Unparseable” denotes an output without valid spatial geometry.

In a separate experiment withholding cribriform supervision, cribriform IoU was 29%, compared with 54% after full concept training, 42% for Gemini 3.1 and 0% for GPT-5.4 and Opus 4.7 (Supplementary Figure S13). The retained performance indicates some transfer to this withheld concept, while the difference from full training shows the additional value of concept-specific examples. Two qualitative comparisons illustrate this variation at the level of individual fields (Supplementary Figure S14).

![](images/42163270c1121d94678176f8582ed01e79254b533e8a864b8e9866d89f188e4f.jpg)  
Supplementary Figure S13. Localization after withholding cribriform supervision. Cribriform IoU for a localizer trained without cribriform examples, the localizer trained with the full concept set, and three frontier baselines. Labels above the bars give IoU percentages.

![](images/8dd88f9bb409bb1809c9c28e2f28f3e8f96664b3e28b7a120cf990e5fa9df09d.jpg)  
Supplementary Figure S14. Cribriform localization with and without direct cribriform supervision. Two prostate fields compare the reference region, PathCowork trained on the full concept set, PathCowork trained with cribriform examples withheld, and three frontier models. Values are example-level IoU percentages. The two fields illustrate variation in transfer to the withheld concept; aggregate results are reported in Supplementary Figure S13. FOV, field of view.

## Supplementary Note 7 | Model and deployment choices

Complementary development experiments examined the trade-offs underlying model and deployment choices: localization across model sizes, resource use at reduced precision and training behavior over one or two epochs. The settings for each experiment are given below.

Backbone comparisons assessed dermoepidermal-junction localization using Dice overlap. The four-model comparison includes Qwen3.5-4B, Qwen3.6-35B-A3B, Qwen3.5-27B and Kimi K2.6 (Supplementary Figure S15a). Mean Dice was approximately 47–48% for the smaller dense and mixture-of-experts models and 51–52% for Qwen3.5-27B and Kimi K2.6. The same measurements are plotted against total and active parameter counts to compare these dense and mixture-of-experts backbones; architecture and parameter count vary together.

A separate 27B dermatopathology experiment compared bf16 server inference with 8-bit and 4-bit local inference (Supplementary Figure S15b). Mean dermoepidermal-junction-region Dice was 29%, 35% and 23%, respectively. The local 4-bit model occupied approximately 15 GB, compared with 28 GB for 8-bit, and per-image latency decreased from approximately 14 to 10.5 s. Both numerical precision and the execution environment differed across these configurations.

Training loss for Kimi K2.6 continued to decrease during a second epoch, reaching the order of $1 0 ^ { - 3 }$ near the end of the run (Supplementary Figure S15c). The one-epoch trace ended at a higher training loss.

![](images/60cbd157d38da2882d5ff9a3de95d0acff91626355da09c282d5c33b923960f6.jpg)

![](images/4b0d29dd8284887c0b3e57a0aa6483f71c4fb3aace0e038066684776397d7436.jpg)

b Numerical precision and deployment requirements<sub>Quantization: locali</sub>  
![](images/e0582e30598f4c34588c9c0a7611214d7eb5c151f6b20dcd2e2c8aac58f1ce1b.jpg)

![](images/b042bf9342583bee43bba822689aa35c76aadc842bba827b90cf05d8d1357b48.jpg)

## c Training duration: Kimi K2.6

![](images/5d32865d7001766c8cbb24c2a80b0b6250c3072ee545d81a37bd45af3874e3f3.jpg)

Supplementary Figure S15. Development studies of model and deployment choices. a, DEJ localization versus total and active parameter counts, showing the same measurements on two logarithmic axes. Filled markers denote dense models and open markers mixture-of-experts models. b, DEJ-region Dice under bf16 server inference and 8-bit and 4-bit local inference (left); model size and per-image latency for the local configurations (right). c, Kimi K2.6 training cross-entropy over one or two epochs. Thin lines show step-level loss and thick lines its exponential moving average (weight 0.1 on the current step). DEJ, dermoepidermal junction.

## Supplementary Note 8 | Spatial formats and reinforcement learning

## Choosing an output format before constructing training targets

The six-format development comparison tested whether Qwen3.5-4B could express a usable spatial response before supervised training. Prompts requested box tokens, a plain polygon, JSON polygons with one hypothetical example, the same JSON format without the example, a clockwise boundary trace, or a segmentation schema containing a box, mask and label. The common 60-tile set was drawn from Ivy GAP<sup>32,33</sup>; selection criteria are described in Methods. Output geometry, region purity and coverage were assessed separately.

The box-token and plain-polygon prompts produced 50 and 53 regions, respectively, all classified as rectangles. Adding one hypothetical JSON example elicited regions on 37 of 60 tiles, with 28 of 37 regions classified as non-rectangular and presence accuracy of 0.75. Removing the example reduced region-producing responses to 5 of 60 tiles and presence accuracy to 0.38 in the archival rerun. Without the example, purity increased from 0.17 to 0.41 and coverage was 0.01. The boundary-trace and segmentation-schema prompts yielded non-rectangular regions in 8 of 26 and 7 of 69 predictions, respectively. We selected JSON with one example because it combined frequent region output, preserved presence discrimination and predominantly non-rectangular geometry.

Conditions 1–3 use preserved original predictions; conditions 4–6 use reruns from 6 September 2026 with reconstructed prompts archived alongside the inputs. The selected JSON example illustrated the response structure for a hypothetical image.

## Effect of supervised initialization

Three atlas-derived glioma experiments compared reinforcement learning from the base model, from a supervised checkpoint, and from a supervised checkpoint trained with additional lower-grade glioma (LGG) negatives. Each condition used one training run; the protocol and reward are specified in Methods.

The base and supervised-initialized runs used the same 2,553-record pool. LGG augmentation affected both supervised initialization and the reinforcement-learning pool; the corresponding supervised validation sets contained 77 records before augmentation and 117 afterward.

Between the first and final validation checks, reward decreased from 0.478 to 0.306 and presence accuracy from 0.75 to 0.58 when reinforcement learning began from the base model. With supervised initialization, validation reward increased from 0.512 to 0.573 and presence accuracy remained near 0.83–0.84. The run with additional LGG negatives reached a validation reward of 0.665 and presence accuracy of approximately 0.89. These results motivated supervised initialization and joint monitoring of localization and presence discrimination.

Supervised logs track response-token negative log-likelihood on training and validation sets. Reinforcement-learning logs track mean reward from 12 prompts per step, alongside greedy-validation reward and presence accuracy at four evaluation events.

## Supplementary Tables

Supplementary Table S1. Datasets represented in the recording collection. Image sources include TCGA<sup>25</sup>, DROID-skin<sup>26,27</sup> and LNCO2<sup>28</sup>; behavioral recordings were acquired through the study platform.

<table><tr><td>Dataset</td><td>Tissue or subspecialty</td><td>Source type</td></tr><tr><td>TCGA-COAD</td><td>Colorectal pathology</td><td>Public</td></tr><tr><td>TCGA-STAD</td><td>Gastric pathology</td><td>Public</td></tr><tr><td>TCGA-BLCA</td><td>Bladder pathology</td><td>Public</td></tr><tr><td>TCGA-PRAD</td><td>Prostate pathology</td><td>Public</td></tr><tr><td>TCGA-DLBC</td><td>Hematopathology</td><td>Public</td></tr><tr><td>TCGA-GBM</td><td>Neuropathology</td><td>Public</td></tr><tr><td>DROID-skin</td><td>Dermatopathology</td><td>Public</td></tr><tr><td>LNCO2</td><td>Colorectal lymph-node pathology</td><td>Public</td></tr><tr><td>Cytology teaching set</td><td>Cytopathology</td><td>Institutional</td></tr></table>

Supplementary Table S2. Intermediate record fields. Coordinates and timestamps link derived examples to the original slide and recording.

<table><tr><td>Field</td><td>Representation</td><td>Content</td></tr><tr><td>Image</td><td>Crop and WSI metadata</td><td>Viewed tissue region, slide identifier, coordinates and magnification</td></tr><tr><td>Narration</td><td>Text and timestamps</td><td>Transcribed utterance, neighboring speech and its time interval</td></tr><tr><td>Grounding</td><td>Spatial coordinates</td><td>Polygon, bounding box or polyline associated with a narrated concept</td></tr><tr><td>Trajectory</td><td>Ordered event records</td><td>Sequence of views, positions, magnifications and transitions</td></tr><tr><td>Revisit</td><td>Linked event records</td><td>Previous observations of the same WSI region</td></tr><tr><td>Derived context</td><td>Structured text</td><td>Normalized description, case summary and interpretation of the narration</td></tr><tr><td>Video</td><td>Session recording</td><td>Visual reference for quality review and event alignment</td></tr></table>

Supplementary Table S3. Model configurations by analysis. Backbone identifiers and training conditions are listed for the individual experiments.
<table><tr><td>Analysis</td><td>Model or configuration</td><td>Role</td></tr><tr><td>Main grounding comparison</td><td>Kimi K2.6; Qwen3.6-27B</td><td>PathCowork-1T and PathCowork-27B; LoRA rank 32, peak learning rate  $1 0 ^ { - 4 }$  , batch size</td></tr><tr><td>Dermatopathology skill</td><td>PathCowork-1T (Kimi K2.6)</td><td>16, two epochs; final checkpoint Adapted concept localizer evaluated in Figure 2e,f and used by the agent in Figure 3</td></tr><tr><td>Brain agent comparison</td><td>PathCowork v4.4 High; Codex GPT-5.6-sol</td><td>Same 30 cases; Claude Code also evaluated, with two missing reports counted as incorrect</td></tr><tr><td>PathCowork-general</td><td>Qwen3.5-4B; LoRA rank 32</td><td>Supervised general navigator; 29 decision slots (Note 3)</td></tr><tr><td>Small-model development</td><td>Qwen3.5-4B; LoRA rank 8</td><td>SFT and subsequent appearance-aware RL experiment (Note 1)</td></tr><tr><td>Glioma format and RL recipe</td><td>Qwen3.5-4B; LoRA rank 32</td><td>Prompt-format selection and base versus SFT-initialized RL (Note 8)</td></tr><tr><td>Prostate data scaling</td><td>Qwen3.5-27B; two epochs</td><td>Increasing training-set size and concept-level evaluation (Note 6)</td></tr><tr><td>Model-size comparison</td><td>Qwen3.5-4B; Qwen3.6-35B-A3B; Qwen3.5-27B; Kimi K2.6</td><td>DEJ localization versus total and active parameter counts</td></tr><tr><td>Quantization comparison</td><td>27B skin localizer; bf16, 8-bit, 4-bit</td><td>Localization, model footprint and latency</td></tr><tr><td>Training-duration comparison Question-source comparison</td><td>Kimi K2.6; one or two epochs</td><td>Supervised training-loss traces</td></tr><tr><td></td><td>Claude / GPT; Haiku 4.5, Sonnet 5, Opus 4.8</td><td>Image-input and separate three-model source comparisons in Figure 5b</td></tr><tr><td>Benchmark categories</td><td>Opus 4.8; Haiku 4.5; GPT-5.5; GPT-5.6-sol; GPT-5.4-mini; Gemini 3.1</td><td>Four categories, with one resident reference</td></tr><tr><td>MIL feature encoders</td><td>UNI2-h; Virchow2; CONCH</td><td>Frozen tile embeddings for slide-level grading</td></tr></table>

Supplementary Table S4. Sources of cases used in agent evaluation. All three agents were assigned the same cases within each task. Evaluation cases were unseen during grounding training and skill construction.
<table><tr><td>Task</td><td>Source</td><td>Evaluation set</td><td>Clinical endpoint</td></tr><tr><td>Skin</td><td>HISTAI38</td><td>20 cases; 69 WSIs</td><td>Melanoma versus dysplastic nevus</td></tr><tr><td>Prostate</td><td>PANDA39</td><td>20 slides</td><td>Cancer detection and ISUP grade</td></tr><tr><td>Breast</td><td>HISTAI38</td><td>39 cases; grades 1/2/3: 10/14/15</td><td>Nottingham grade and component scores</td></tr><tr><td>Brain</td><td>TCGA-GBM; TCGA-LGG25</td><td>30 cases; 15 per class</td><td>GBM versus lower-grade glioma</td></tr><tr><td>Colorectal lymph nodes</td><td>LNCO228</td><td>77 slides; 10 positive, 67 negative</td><td>Node detection and metastasis classification</td></tr></table>

Supplementary Table S5. Paired comparisons of CoworkMIL with ABMIL. Means and mean paired differences in quadratic-weighted κ are based on 20 matched training seeds per setting, evaluated on the same 30-slide test set. N is the number of training slides. Two-sided exact Wilcoxon signed-rank tests use the archived seed-level scores, stored to four decimal places; none of these contrasts has a zero difference or tied absolute differences. All 15 comparisons are reported; $^ { \star } P < 0 . 0 5$ $^ { \star \star } P < 0 . 0 1$ . Positive differences favor CoworkMIL.
<table><tr><td>Encoder</td><td>N</td><td>CoworkMIL</td><td>ABMIL</td><td>Difference</td><td> $P$ </td></tr><tr><td>UNI2-h</td><td>10</td><td>0.321</td><td>0.279</td><td>+0.041</td><td>0.153646</td></tr><tr><td>UNI2-h</td><td>20</td><td>0.544</td><td>0.466</td><td>+0.078</td><td>0.001017**</td></tr><tr><td>UNI2-h</td><td>30</td><td>0.620</td><td>0.552</td><td>+0.067</td><td>0.029575*</td></tr><tr><td>UNI2-h</td><td>40</td><td>0.690</td><td>0.660</td><td>+0.030</td><td>0.230513</td></tr><tr><td>UNI2-h</td><td>50</td><td>0.712</td><td>0.724</td><td>-0.012</td><td>0.348810</td></tr><tr><td>Virchow2</td><td>10</td><td>0.376</td><td>0.261</td><td>+0.115</td><td>0.010689*</td></tr><tr><td>Virchow2</td><td>20</td><td>0.486</td><td>0.385</td><td>+0.101</td><td>0.009436* **</td></tr><tr><td>Virchow2</td><td>30</td><td>0.620</td><td>0.469</td><td>+0.151</td><td>0.001432 **</td></tr><tr><td>Virchow2</td><td>40</td><td>0.731</td><td>0.611</td><td>+0.120</td><td>0.001432**</td></tr><tr><td>Virchow2</td><td>50</td><td>0.773</td><td>0.761</td><td>+0.013</td><td>0.621513</td></tr><tr><td>CONCH</td><td>10</td><td>0.347</td><td>0.271</td><td>+0.076</td><td>0.032768*</td></tr><tr><td>CONCH</td><td>20</td><td>0.585</td><td>0.484</td><td>+0.101</td><td>0.017181*</td></tr><tr><td>CONCH</td><td>30</td><td>0.718</td><td>0.606</td><td>+0.112</td><td>0.001209**</td></tr><tr><td>CONCH</td><td>40</td><td>0.742</td><td>0.662</td><td>+0.081</td><td>0.003153 **</td></tr><tr><td>CONCH</td><td>50</td><td>0.741</td><td>0.643</td><td>+0.098</td><td>0.000168* **</td></tr></table>

Supplementary Table S6. EAGLE and MIL pipelines on matched training subsets and test slides. All rows use the same 20 subsets of 20 training slides and the same 30-slide PANDA test cohort. Quadratic-weighted κ is reported as mean ± sample standard deviation across subsets; MAE is the mean absolute error in ISUP grade. EAGLE uses frozen CTransPath–CHIEF tile selection<sup>42,43</sup>, frozen Virchow2 class-token features, mean pooling and a fitted logistic-regression classifier. Its indicated resolution applies to both tile selection and re-encoding. The MIL rows use frozen Virchow2 class-token plus mean patch-token features from native level-0 tiles and a trained attention aggregator; CoworkMIL additionally uses grounding-derived training supervision.

<table><tr><td>Pipeline</td><td> $\kappa ( \mathsf { m e a n } \pm \mathsf { s . d . } )$ </td><td>Mean MAE</td></tr><tr><td>EAGLE, 0.5 µm per pixel</td><td> $0 . 4 3 0 \pm 0 . 1 6 0$ </td><td>1.225</td></tr><tr><td>EAGLE, 2.0 µm per pixel</td><td> $0 . 4 3 5 \pm 0 . 1 0 8$ </td><td>1.298</td></tr><tr><td>ABMIL, Virchow2</td><td> $0 . 3 8 5 \pm 0 . 1 3 3$ </td><td>1.347</td></tr><tr><td>CoworkMIL, Virchow2</td><td> $0 . 4 8 6 \pm 0 . 1 2 0$ </td><td>1.178</td></tr></table>

Supplementary Table S7. Composition of the retained recorded-review question bank. Counts refer to unique retained question identifiers in the two source-bank exports. Environmental sensing comprises two subtypes. The separate 100-question source comparisons and 148-question paired image-input analysis use the subsets described in Methods.

<table><tr><td>Category</td><td>Content</td><td>Questions</td></tr><tr><td>Rejection of an inapplicable premise</td><td>Finding or premise verification</td><td>128</td></tr><tr><td>Multi-slide reasoning</td><td>Evidence across fields or slides</td><td>62</td></tr><tr><td>Goal-directed navigation</td><td>Next region to inspect</td><td>59</td></tr><tr><td>Environmental sensing</td><td>Viewing scale</td><td>63</td></tr><tr><td></td><td>Tissue location</td><td>59</td></tr><tr><td colspan="2">Total</td><td>371</td></tr></table>

## Supplementary Video

## Supplementary Video 3 | General navigation learned from pathologist behavior.

Demonstration of PathCowork-general navigating a CAMELYON17 lymph-node slide using a learned viewing policy. The main panel shows the current field and candidate grid cells; the overview shows the viewing trajectory and spatial context. The navigator successively zooms into selected fields, inspects tissue at higher magnification and returns to a broader view before visiting another region. The status panel reports actions, model outputs and measured inference times. Blue overlays indicate reference lesion annotations (filled where read), yellow lines indicate the viewing trajectory and the red box indicates the current field. This recorded example contains 10 steps, with 3.41 s total model inference time and 3.68 s recorded runtime. The supplied visualization lasts 17.9 s, including display holds; its playback timing is preserved. The submission video adds opening and closing title cards and has no audio.