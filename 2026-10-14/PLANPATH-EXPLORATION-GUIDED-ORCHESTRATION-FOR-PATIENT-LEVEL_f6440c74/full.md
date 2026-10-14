# PLANPATH: EXPLORATION-GUIDED ORCHESTRATION FOR PATIENT-LEVEL PATHOLOGY REASONING

Jiusong Ge<sup>1</sup>, Mingzhao He<sup>1</sup>, Qidong Liu<sup>1,∗</sup>, Chunbao Wang<sup>1</sup>, Chunze Yang<sup>1</sup>, Di Zhang<sup>1</sup> Jiashuai Liu<sup>1</sup>, Mireia Crispin-Ortuzar<sup>2</sup>, Huazhu Fu<sup>3</sup>, Chen Li<sup>4,1,∗</sup>, Zeyu Gao<sup>2,∗</sup>

<sup>1</sup>Xi’an Jiaotong University <sup>2</sup>University of Cambridge <sup>3</sup>A\*STAR <sup>4</sup>Chinese Academy of Medical Sciences <sup>∗</sup>Corresponding author

## ABSTRACT

A patient’s pathology case spans tens of slides from distinct specimens and anatomical sites, each carrying non-redundant, site-specific evidence. Forming a patient-level conclusion therefore requires locating the slides relevant to a given question and integrating complementary findings across them. However, existing pathology agents analyze one slide at a time and describe morphology rather than measure it, leaving evidence acquisition across a patient’s specimens and patientlevel quantification underexplored. To address this gap, we propose PlanPath, an agentic framework for patient-level pathology visual question answering (VQA). PlanPath turns the question into an exploration plan that specifies the evidence required, routes to the slides likely to carry it, and drives coarse-to-fine acquisition from tissue architecture down to individual cells. Tissue- and cell-level tools supply the counts, proportions, and extents, which PlanPath aggregates across slides into a traceable answer. To enable evaluation, we build two patient-level VQA datasets for renal and gastric cancer, averaging 15 WSIs per patient, with pathologist-verified question–answer pairs derived from diagnostic reports. Experimental results show that PlanPath outperforms the strongest baseline by 9.4% and 12.1% in overall accuracy on the two datasets and further improves existing slide-level agents when plugged into our framework. The dataset and code are available at: https://github.com/JSGe-AI/PlanPath\_1.

## 1 INTRODUCTION

Recent advances in pathology foundation models and vision–language models have improved morphological representation and image–text alignment in computational pathology, providing reusable technical foundations for whole-slide image (WSI) understanding (Chen et al., 2024a; Lu et al., 2024a). Building upon these capabilities, pathology agents powered by large language models (LLMs) can autonomously localize relevant regions, invoke specialized analysis tools, and integrate diagnostic findings in response to clinical questions (Chen et al., 2025a; Ghezloo et al., 2025). These approaches have demonstrated promising performance on slide-level tasks while making the visual evidence behind their conclusions more explicit.

However, to better translate these advances into clinical diagnostic practice, pathology analysis should move beyond individual slides toward the patient level. Such analysis can integrate complementary evidence distributed across distinct specimens and anatomical sites, thereby providing a more complete and clinically representative characterization of a patient’s pathological state. Nevertheless, this setting remains relatively underexplored in existing studies. To bridge this gap, we formally define patient-level pathology visual question answering (VQA) as the task of generating a patient-level answer with explicit visual grounding from a patient’s collection of slides and a diagnostic question through cross-slide, multiscale evidence acquisition.

Despite their effectiveness in slide-level analysis, existing pathology agents (Sun et al., 2026; Lyu et al., 2025) remain limited for patient-level tasks. A naive extension is to aggregate answers independently generated from individual slides. However, such aggregation has a fundamental limitation: it neither selects question-relevant slides nor integrates complementary evidence across them (Fig. 1b). This is critical because different patient-level questions often have different evidence requirements, while the relevant evidence is unevenly distributed across a patient’s slide collection (Koyun et al., 2026; Huang et al., 2026). Effective patient-level analysis therefore requires first identifying slides relevant to the given question and then integrating complementary evidence across them, rather than treating individual slides as independent units of analysis (Fig. 1c).

![](images/84a1b2138c580ca950b826964ef2d846baf3ce88cb60d1d88538ccb2b003abf7.jpg)  
Figure 1: Comparison of slide-level and patient-level pathology VQA. (a) A slide-level agent performs well on single-slide questions. (b) Directly applying slide-level agents to multiple slides lacks effective slide selection and cross-slide reasoning. (c) PlanPath addresses patient-level questions through exploration planning, slide ranking, and cross-slide evidence integration.

Beyond slide selection, patient-level diagnosis often relies not only on the presence of pathological findings, but also on their quantitative characterization, including their amount, proportion, extent, and distribution. For example, the Gleason score of prostate cancer is determined by the relative proportions of its predominant and secondary patterns (Van Leenders et al., 2020). Existing pathology agents (Sun et al., 2026; Yang et al., 2026) reason over a limited set of local observations; while they can recognize local morphology, they struggle to translate it into reliable patient-level quantitative conclusions. Effective patient-level analysis therefore also requires quantifying findings and aggregating them across slides at appropriate spatial scales.

To address these challenges, we propose PlanPath, a pathology agentic framework for patientlevel visual question answering. Given a patient-level question, PlanPath formulates an exploration plan that specifies the required diagnostic evidence and guides question-driven slide selection and cross-slide exploration. It then performs hierarchical cross-slide evidence acquisition, characterizing question-relevant regions through morphological observations, while tissue-level analysis provides spatial context and cell-level quantification is invoked when quantitative evidence is required. These complementary findings are integrated across slides at the patient level, with tissue- and celllevel analyses supporting measurements of areas, counts, proportions, and cross-slide distributions for questions that demand quantitative reasoning. Notably, as a general patient-level orchestration framework, PlanPath can adapt existing slide-level pathology agents to patient-level tasks.

Moreover, to support systematic evaluation, we collaborate with pathologists to construct two patient-level pathology VQA datasets for renal and gastric cancer. Questions and reference answers are derived from original diagnostic records and available slide evidence, with explicit verification that the anatomical sites required to answer each question are supported by corresponding slides.

The main contributions of this work are summarized as follows:

• We first formally define patient-level pathology VQA as a distinct task that requires reasoning over a patient’s collection of slides rather than individual slides. To tackle this task, we propose PlanPath, a patient-level pathology agentic framework for question-driven slide selection, crossslide evidence acquisition, and complementary evidence integration.

• We develop a multiscale evidence acquisition and quantitative analysis strategy that integrates slide-level context, tissue-level structure, and cell-level measurements, enabling aggregation of evidence for patient-level quantitative reasoning beyond isolated local observations.

• We construct two patient-level pathology VQA datasets for renal and gastric cancer, comprising 3,245 WSIs from 213 patients and 1,623 pathologist-verified question–answer pairs, and use them as standardized benchmarks to systematically evaluate PlanPath and a broad range of existing methods under the patient-level setting.

## 2 DATASET CONSTRUCTION AND PROBLEM FORMULATION

This section presents the construction and quality control procedures for two patient-level pathology visual question answering (PatientVQA) datasets, analyzes their scale, question types, and diagnostic content, and formally defines the task inputs, outputs, and objectives.

## 2.1 PATIENT-LEVEL PATHOLOGY VQA DATASET CONSTRUCTION

We construct PatientVQA-Gastric and PatientVQA-Renal using real-world pathology records from patients with renal and gastric cancer. We include patients with definitive diagnoses, collect the corresponding pathology reports, and organize WSIs from different sampling sites into patient-level slide collections. GPT-5.6 Sol is used to derive candidate question–answer pairs from the reports, with reference answers grounded in explicitly documented diagnostic findings. The questions cover tumor type, grade, local invasion, margin status, lymph node involvement, and pathological staging. For quality control, two pathologists cross-review the candidate pairs and exclude those with medical or logical errors, ambiguous wording, insufficient support from the reports, or inadequate visual evidence in the available slides. The retained pairs are linked to their corresponding patient-level slide collections to form the final datasets.

## 2.2 DATASET STATISTICS AND ANALYSIS

The two datasets comprise 213 patients, 3,245 WSIs, and 1,623 question–answer pairs. PatientVQA Gastric contains 104 patients, 2,227 WSIs, and 923 question–answer pairs, while PatientVQA-Renal contains 109 patients, 1,018 WSIs, and 700 question–answer pairs. The average numbers of slides per patient are 21.41 and 9.34, with ranges of 13–58 and 2–36, respectively, enabling evaluation across patient-level slide collections of varying sizes.

Both datasets include yes/no, single-answer multiple-choice (MCQ), and short-answer (Open) questions, accounting for 29.0%, 28.1%, and 42.9% of PatientVQA-Gastric, and 46.7%, 34.6%, and 18.7% of PatientVQA-Renal, respectively. These question types cover the presence of pathological findings, predefined diagnostic categories, and diagnostic answers. The diagnostic content covers tumor type, grade, local invasion, margin status, and pathological staging. Each patient is associated with multiple questions with varying slides and evidence requirements, supporting evaluation of morphological recognition, question-driven slide selection, and cross-slide evidence integration.

## 2.3 PROBLEM FORMULATION OF PATIENT-LEVEL VQA

Let $\boldsymbol { S _ { p } } = \{ W _ { p , i } \} _ { i = 1 } ^ { N _ { p } }$ denote the WSI collection of patient $p .$ . Given a diagnostic question q, patientlevel pathology VQA aims to acquire evidence from $S _ { p }$ and generate a visually grounded answer:

$$
( \hat { a } , \mathcal { E } ) = \mathcal { F } ( S _ { p } , q ) ,\tag{1}
$$

where $\mathcal { F }$ is the answering system, aˆ the predicted answer, and $\mathcal { E }$ the supporting evidence.

The evidence set $\mathcal { E }$ contains question-relevant visual observations and, when needed, quantitative measurements, each linked to its source slide, spatial region, and observation scale. Depending on the question, evidence may be acquired from one or multiple WSIs.

The task therefore requires selecting relevant slides and regions, acquiring evidence at appropriate scales, and integrating it into an accurate, evidence-grounded patient-level answer.

![](images/8ebc84d9ef8a5bc3feffcc0d8c4dcb22b5a654b20a79fa3f3a9752a6701ba887.jpg)  
Figure 2: Overview of PlanPath. The Agent Brain plans and coordinates question-driven slide retrieval and cross-slide multiscale evidence acquisition, followed by iterative evidence integration for prediction, confidence estimation, and traceable reasoning.

## 3 METHOD

## 3.1 FRAMEWORK OVERVIEW

We propose PlanPath, an agentic framework for patient-level pathology VQA. As illustrated in Fig. 2, a unified Agent Brain coordinates the Slide Router, cross-slide navigation, and analysis tools: the Tissue-level Analysis Tool, Cell-level Quantification Tool, and Image Description Tool.

The Agent Brain first constructs an exploration plan that specifies the required evidence, analysis scales, and stopping criteria, denoted as Question-driven Exploration Planning. Guided by this plan, the Slide Router retrieves relevant slides through Patient-level Slide Routing. PlanPath then performs Hierarchical Evidence Acquisition by progressively exploring candidate ROIs across slides and magnifications and invoking analysis tools as needed. All acquired evidence is stored with its source slide, ROI, magnification, and spatial lineage, enabling the patient-level Evidence Integration and Answer Generation to remain traceable to visual and quantitative evidence.

## 3.2 QUESTION-DRIVEN EXPLORATION PLANNING

PlanPath first transforms the original question q into a structured exploration plan that guides subsequent evidence acquisition:

$$
\Pi ( q ) = \left( \mathcal { D } , \tau , \mathcal { T } , \mathcal { R } , \mathcal { G } \right) ,\tag{2}
$$

where D denotes a set of semantic descriptions used for slide retrieval, τ represents the logical property of the question, T specifies the target evidence, R defines the required spatial resolutions and tool calls, and G specifies evidence sufficiency and stopping criteria.

According to their reasoning requirements, input questions are categorized into existential, exhaustive, extremum, quantitative, or composite types. These categories determine the required evidence coverage. For example, an existential question may terminate once reliable direct positive evidence is obtained, whereas patient-level negative conclusions, extrema, extent-related assessments, and other global judgments generally require broader slide coverage. The derived exploration plan also serves as a tool-routing mechanism. Based on Π, the Agent Brain determines which analysis tools are required for the current question and specifies their roles in evidence acquisition. Tools irrele vant to the current question are excluded from the execution path. In conclusion, PlanPath does not apply a fixed analysis pipeline to every slide. Instead, it first determines what evidence is required and decides where, at what scale, and with which tools to acquire it.

## 3.3 PATIENT-LEVEL SLIDE ROUTING

The Slide Router identifies and prioritizes slides that are likely to contain evidence relevant to the current question. Given the patient WSI collection S and the retrieval descriptions D specified in the exploration plan Π, it produces a relevance-ranked candidate slide pool for evidence acquisition. The Slide Router is used only to determine the order and scope of exploration. Routing scores are not considered diagnostic evidence and do not directly contribute to the final answer. Any retrieved slide must undergo ROI-level analysis before it can support a patient-level conclusion.

Candidate Pool Initilization. For the patient WSI collection $s ,$ the Slide Router evaluates the relevance of each slide according to the retrieval descriptions D defined in the exploration plan. For slide $S _ { i } ,$ its routing score is computed as:

$$
r _ { i } = \operatorname* { m a x } _ { d \in \mathcal { D } } \sin \left( f _ { \mathrm { s l i d e } } ( S _ { i } ) , f _ { \mathrm { t e x t } } ( d ) \right) ,\tag{3}
$$

where $f _ { \mathrm { s l i d e } } ( \cdot )$ and $f _ { \mathrm { t e x t } } ( \cdot )$ are the slide/text encoders, and sim(·, ·) denotes cross-modal similarity. Then, the slides are ranked according to $r _ { i } ,$ and an initial candidate pool is constructed as:

$$
\mathcal { C } _ { 0 } = \mathrm { T o p K } _ { S _ { i } \in \mathcal { S } } \left( r _ { i } \right) .\tag{4}
$$

Using multiple complementary retrieval descriptions allows slides containing different but relevant morphological patterns to enter the candidate pool.

Candidate Pool Extension. Importantly, $\mathcal { C } _ { 0 }$ is an initial candidate pool rather than a fixed retrieval set. If all slides in the current pool have been explored but the required patient-level evidence remains insufficient, the unmet evidence requirements are fed back to the Slide Router. The router then retrieves additional slides from the remaining WSI collection:

$$
\mathcal { C } _ { t + 1 } = \mathcal { C } _ { t } \cup \mathrm { T o p K } \left( S \setminus \mathcal { C } _ { t } \right) ,\tag{5}
$$

where retrieval is conditioned on the updated evidence requirements. The retrieved slides are appended to the candidate pool and undergo the hierarchical evidence acquisition procedure.

Notably, the extent of candidate coverage is also determined by the question type τ . For focal existential findings, PlanPath prioritizes high-ranking slides and may terminate once sufficient positive evidence is obtained. For negative, exhaustive, extremum, or extent-related tasks, broader coverage is maintained to reduce the risk of missing critical evidence that appears only on a subset of slides.

## 3.4 HIERARCHICAL EVIDENCE ACQUISITION

For each candidate slide, PlanPath adopts a coarse-to-fine strategy to progressively localize, inspect, and quantify question-relevant pathological evidence across multiple spatial scales. Specifically, evidence acquisition is organized around three complementary components: tissue context, morphological observation, and cell-level quantification.

Coarse-to-Fine ROI Selection. PlanPath first establishes global tissue context from lowmagnification views and identifies candidate ROIs according to the exploration plan. Each candidate ROI R is prioritized based on semantic relevance, tissue context, and image quality:

$$
\begin{array} { r } { s ( R ) = \alpha s _ { \mathrm { s e m } } ( R ) + \beta s _ { \mathrm { t i s s u e } } ( R ) + \gamma s _ { \mathrm { q u a l i t y } } ( R ) , } \end{array}\tag{6}
$$

where the three terms measure question-related semantic relevance, consistency with target tissue structures, and image quality, respectively. High-priority ROIs are progressively examined at higher magnifications while preserving their parent–child spatial relationships.

Tissue-Level Analysis. When tissue composition, lesion extent, or spatial interfaces are relevant to the question, the Tissue-level Analysis Tool extracts tissue regions, areas, distributions, and interfaces from the WSI. These outputs provide global spatial context and guide subsequent ROI selection rather than directly determining the diagnosis. For quantitative tasks, tissue-level measurements can further support the estimation of areas, proportions, and spatial distributions.

Morphological Observation. For selected ROIs, the Image Description Tool generates morphological observations at the corresponding magnification. Low-magnification views characterize tissue architecture and spatial organization, whereas higher-magnification views capture cellular arrangement and fine-grained morphology. To separate visual observation from diagnostic reasoning, the tool describes the visible pathological findings rather than directly predicting the final answer.

Cell-Level Quantification. When quantitative evidence is required, the Cell-level Quantification Tool analyzes selected high-magnification ROIs to obtain cell counts, proportions, densities, and spatial statistics. Quantification is restricted to ROIs satisfying relevance, image-quality, and tissuecoverage requirements, with each measurement linked to its source slide and ROI.

Evidence Recording and Iterative Exploration. Each acquired observation is stored as a structured evidence record:

$$
e _ { j } = ( S _ { j } , R _ { j } , m _ { j } , v _ { j } , z _ { j } , \ell _ { j } ) ,\tag{7}
$$

where $S _ { j } , R _ { j } , m _ { j } , v _ { j } , z _ { j }$ , and $\ell _ { j }$ denote the source slide, ROI, magnification, morphological observation, optional quantitative measurement, and spatial lineage, respectively.

After each step, the Agent Brain evaluates evidence and selects the next action:

$$
\begin{array} { r l r } { a _ { t } = \pi \left( q , \Pi , \mathcal { E } _ { t } , \mathcal { C } _ { t } \right) , } & { { } } & { a _ { t } \in \{ \mathrm { s t o p } , \mathrm { z o o m } , \mathrm { n e x t - r o i } , \mathrm { n e x t - s l i d e } , \mathrm { r e r o u t e } \} } \end{array}\tag{8}
$$

Here, $\mathcal { E } _ { t }$ denotes evidence accumulated by step $t ,$ and $\mathcal { C } _ { t }$ the candidate slide pool. These actions determine whether PlanPath stops, zooms into the ROI, moves to another ROI or slide, or returns to the Slide Router for additional slides when the pool is insufficient. This process allows evidence acquisition to adapt to observed evidence rather than follow a fixed traversal of all slides and ROIs.

Cross-Scale Quantitative Aggregation. For questions requiring patient-level quantitative reasoning, valid measurements obtained from tissue- and cell-level analyses are aggregated according to the quantitative requirement of the question. Let $z _ { i }$ denote the valid measurements obtained from slide $S _ { i }$ . Patient-level quantitative evidence is computed as:

$$
Z _ { p } = \mathrm { A g g } _ { \tau } \left( \{ z _ { i } \} _ { i \in \mathcal { C } _ { \mathrm { c o v } } } \right) ,\tag{9}
$$

where $\mathcal { C } _ { \mathrm { c o v } }$ denotes the set of slides with valid measurements. Depending on the task, the aggregation may correspond to counting, proportion estimation, extremum selection, or preservation of cross-slide distributions. Overlapping ROIs are excluded or corrected to avoid double counting, and numerical conclusions are reported only when predefined coverage and quality requirements are satisfied. Through this hierarchical process, PlanPath links tissue context, regional morphology, and cell-level measurements into a unified and traceable evidence chain for patient-level reasoning.

## 3.5 EVIDENCE INTEGRATION AND ANSWER GENERATION

All visual observations and quantitative measurements are stored in a patient-level evidence repository $\mathcal { E } _ { t } = \{ e _ { j } \}$ . Rather than aggregating independent slide-level predictions, PlanPath integrates complementary cross-slide evidence according to the exploration plan. Before termination, an Evidence Gate evaluates whether the accumulated evidence satisfies the requirements defined by Π:

$$
G \left( \mathcal { E } _ { t } \mid \Pi \right) = \mathbf { 1 } \left\{ C _ { t } \geq C _ { \operatorname* { m i n } } \wedge Q _ { t } \geq Q _ { \operatorname* { m i n } } \wedge T _ { t } = 1 \wedge L _ { t } = 1 \right\} ,\tag{10}
$$

where $C _ { t } , Q _ { t } , T _ { t }$ , and $L _ { t }$ denote coverage, quality, target satisfaction, and spatial consistency.

If the Evidence Gate is not satisfied, PlanPath continues the iterative exploration in Sec. 3.4 according to the remaining evidence requirements until sufficient evidence is obtained or the available slides or analysis budget are exhausted. Once satisfied, the Agent Brain constructs a compact evidence context $\mathcal { E } ^ { * }$ and generates the final answer:

$$
\hat { y } = \mathrm { D e c } \left( q , \Pi , \mathcal { E } ^ { * } \right) .\tag{11}
$$

The output follows the original question format and retains confidence, evidence sufficiency, and traceable evidence identifiers. If exploration ends before the requirements are fully satisfied, the answer is marked as low-confidence and evidence-insufficient.

## 4 EXPERIMENT

## 4.1 EXPERIMENTAL SETUP

Implementation Details. The Agent Brain in PlanPath is instantiated with Qwen3.5-4B (Qwen Team, 2026), while the Slide Router is implemented using TITAN (Ding et al., 2025). The Tissuelevel Analysis Tool employs a prototype-guided tissue segmentation model, the Cell-level Quantification Tool uses HistoPLUS (Adjadj et al., 2026), and the Image Description Tool is based on Patho-R1-7B (Zhang et al., 2026). Cross-slide navigation is jointly driven by the tissue-level analysis results and PLIP (Huang et al., 2023): tissue maps provide spatial context, while PLIP evaluates the semantic relevance between candidate regions and the current evidence requirements. Together, these components support slide selection, multiscale evidence acquisition, and patient-level evidence integration. Details of each component are provided in Appendix B.

Table 1: Performance comparison on PatientVQA-Gastric and PatientVQA-Renal. The best result in each column is shown in bold, and the second-best result is underlined.
<table><tr><td rowspan="3">Method</td><td rowspan="3">Input</td><td colspan="4">PatientVQA-Gastric</td><td colspan="4">PatientVQA-Renal</td></tr><tr><td>Total</td><td>MCQ</td><td>Yes/No</td><td>Open</td><td>Total</td><td>MCQ</td><td>Yes/No</td><td>Open</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9">Pretrained MLLM Models</td></tr><tr><td>GPT-5.5</td><td>Slide-T</td><td>35.10</td><td>39.77</td><td>30.97</td><td>34.85</td><td>40.71</td><td>23.97</td><td>60.55</td><td>22.14</td></tr><tr><td>GPT-5.6-Luna</td><td>Slide-T</td><td>37.70</td><td>36.29</td><td>50.75</td><td>29.80</td><td>49.29</td><td>47.11</td><td>62.08</td><td>21.37</td></tr><tr><td>Qwen3.5-4B</td><td>Slide-T</td><td>50.38</td><td>42.86</td><td>65.30</td><td>45.20</td><td>55.14</td><td>50.00</td><td>69.11</td><td>29.77</td></tr><tr><td>Qwen3.5-9B</td><td>Slide-T</td><td>47.24</td><td>41.70</td><td>62.31</td><td>40.66</td><td>55.57</td><td>46.69</td><td>71.56</td><td>32.06</td></tr><tr><td colspan="10">Supervised Pathology Models</td></tr><tr><td>MedDr</td><td>Slide-T</td><td>30.44</td><td>30.50</td><td>64.18</td><td>7.58</td><td>55.86</td><td>58.26</td><td>70.34</td><td>15.27</td></tr><tr><td>WSI-VQA</td><td>WSI</td><td>32.83</td><td>44.79</td><td>47.39</td><td>15.15</td><td>15.86</td><td>13.22</td><td>20.18</td><td>9.92</td></tr><tr><td>TITAN</td><td>WSI</td><td></td><td>26.64</td><td>39.93</td><td></td><td></td><td>13.64</td><td>57.80</td><td></td></tr><tr><td>Quilt-LLaVA</td><td>Patch</td><td>29.47</td><td>51.35</td><td>35.82</td><td>10.86</td><td>37.00</td><td>61.98</td><td>29.97</td><td>8.40</td></tr><tr><td>WSI-LLaVA</td><td>WSI</td><td>32.50</td><td>22.78</td><td>67.91</td><td>14.90</td><td>49.00</td><td>35.12</td><td>71.87</td><td>17.56</td></tr><tr><td>MedGemma-1.5-4B</td><td>Slide-T</td><td>50.81</td><td>52.12</td><td>64.93</td><td>40.40</td><td>60.57</td><td>59.50</td><td>73.39</td><td>30.53</td></tr><tr><td>SlideChat</td><td>WSI</td><td>34.78</td><td>26.25</td><td>63.43</td><td>20.96</td><td>44.43</td><td>17.77</td><td>71.56</td><td>25.95</td></tr><tr><td colspan="10">Agentic Systems</td></tr><tr><td>PathAgent</td><td>WSI</td><td>33.91</td><td>40.93</td><td>38.81</td><td>26.01</td><td>33.14</td><td>46.28</td><td>30.28</td><td>16.03</td></tr><tr><td>PathNavigate</td><td>WSI</td><td>34.45</td><td>43.24</td><td>58.96</td><td>12.12</td><td>55.14</td><td>57.85</td><td>67.58</td><td>19.08</td></tr><tr><td>GIANT</td><td>WSI</td><td>37.27</td><td>44.40</td><td>33.21</td><td>35.35</td><td>47.14</td><td>34.30</td><td>63.30</td><td>30.53</td></tr><tr><td>BEACON</td><td>WSI</td><td></td><td>33.98</td><td>38.81</td><td></td><td></td><td>47.11</td><td>25.08</td><td></td></tr><tr><td>PlanPath (ours)</td><td>WSI</td><td>60.24</td><td>57.92</td><td>70.15</td><td>55.05</td><td>72.71</td><td>78.51</td><td>74.92</td><td>56.49</td></tr></table>

Baselines. We compare PlanPath with three categories of methods: general-purpose multimodal Large Language Models, including GPT-5.5 (OpenAI, 2026a), GPT-5.6-Luna (OpenAI, 2026b), Qwen3.5-4B (Qwen Team, 2026), and Qwen3.5-9B (Qwen Team, 2026); Supervised Pathology Models, including MedDr (He et al., 2026), WSI-VQA (Chen et al., 2024b), TITAN (Ding et al., 2025), Quilt-LLaVA (Seyfioglu et al., 2024), WSI-LLaVA (Liang et al., 2025), MedGemma-1.5- 4B (Sellergren et al., 2026), and SlideChat (Chen et al., 2025b); and Agentic Systems, including PathAgent (Chen et al., 2025a), PathNavigate (Yang et al., 2026), GIANT (Buckley et al., 2025), and BEACON (Wong et al., 2026). For slide-level methods, each slide is analyzed independently, after which Qwen3.5-4B aggregates the outputs from all slides into a patient-level answer. We evaluate these baselines using a majority-voting aggregation strategy, with results reported in Appendix C.2.

Evaluation Metrics. We use accuracy for MCQ, Yes/No, and open-ended short-answer questions, and report Total Accuracy over all questions as the overall metric for patient-level VQA performance. BLEU-1 and ROUGE-L are reported in Appendix C.1.

## 4.2 MAIN RESULTS ON PATIENT-LEVEL PATHOLOGY VQA

Table 1 presents the main results on PatientVQA-Gastric and PatientVQA-Renal. PlanPath achieves the best overall performance on both benchmarks. On PVQA-Gastric, PlanPath reaches a Total Accuracy of 60.24%, outperforming the strongest baseline by 9.43 percentage points. It also achieves the best performance on MCQ, Yes/No, and open-ended questions. The improvement is particularly notable for open-ended questions, where PlanPath achieves an accuracy of 55.05%. On Renal, the performance improvement is even more pronounced. PlanPath achieves a Total Accuracy of 72.71%, exceeding the strongest baseline by 12.14 percentage points. It reaches an MCQ accuracy of 78.51% and an open-ended accuracy of 56.49%. Overall, these results demonstrate that question-driven slide selection and cross-slide evidence integration are particularly effective for patient-level pathology VQA tasks that require integrating complementary information distributed across multiple WSIs.

![](images/3defd3d1d25489ad1aa8cbda7dd4530f90df2f4ed8cd000aceb8d3ce663e0ce7.jpg)  
Figure 3: Compatibility of PlanPath with slide-level pathology agents.

Table 2: Evaluation of patient-level quantitative questions using exact-match and tolerance-based accuracy. All values are reported as %.
<table><tr><td>Method</td><td></td><td>PathAgent PathNavigate</td><td>GIANT</td><td>PlanPath</td></tr><tr><td>Exact Match</td><td>5.56</td><td>16.67</td><td>11.11</td><td>38.89</td></tr><tr><td>Tol. ±5</td><td>5.56</td><td>16.67</td><td>16.67</td><td>61.11</td></tr><tr><td>Tol. ±10</td><td>11.11</td><td>44.44</td><td>38.89</td><td>72.22</td></tr><tr><td>Tol. ±15</td><td>11.11</td><td>44.44</td><td>38.89</td><td>72.22</td></tr><tr><td>Tol. ±20</td><td>38.89</td><td>55.56</td><td>66.67</td><td>83.33</td></tr></table>

## 4.3 COMPATIBILITY WITH SLIDE-LEVEL AGENTS

We evaluate the compatibility of PlanPath with three representative slide-level pathology agents: PathAgent, PathNavigate, and GIANT. As shown in Fig. 3, integrating PlanPath consistently improves all three agents on both patient-level datasets. On Gastric, the gains range from 7.72 to 8.02 percentage points, while improvements of 3.00–6.15 points are observed on Renal. These consistent gains demonstrate that PlanPath can effectively extend different slide-level agents to patient-level multi-slide reasoning without relying on a specific underlying agent. Detailed performance changes across individual question types are provided in Appendix C.3.

## 4.4 QUANTITATIVE REASONING AND INFERENCE EFFICIENCY

We further evaluate PlanPath in terms of quantitative reasoning and inference efficiency. As shown in Table 2, PlanPath consistently outperforms representative pathology agents under different quantitative evaluation criteria, achieving 38.89% exact match and 61.11%, 72.22%, 72.22%, and 83.33% accuracy under tolerance thresholds of ±5, ±10, ±15, and ±20, respectively.

Meanwhile, Table 3 compares prompt token consumption on the closed-ended question subset common to all methods, including BEACON, for fair comparison. On this cohort, PlanPath requires 164.32K and 120.03K tokens per question on Gastric and Renal, respectively, substantially fewer than PathAgent, PathNavigate, GIANT, and BEACON. These results indicate that question-driven slide routing and hierarchical evidence acquisition reduce unnecessary processing of irrelevant slides and redundant visual information while maintaining patient-level reasoning performance.

## 4.5 ABLATION AND CASE STUDY

Table 4 evaluates key PlanPath components. Removing the Slide Router reduces accuracy from 65.62% to 64.02%, confirming the importance of question-driven slide selection. Removing the Tissue-level Analysis Tool causes the largest drop of 3.57 percentage points, highlighting the role of tissue context in relevant-region exploration. Removing Cell-level Quantification yields a smaller decrease to 65.19%, suggesting a more task-dependent benefit. Removing both tissue- and cell-level tools further degrades performance, supporting the value of multiscale evidence acquisition.

Figure 4 presents a quantitative VQA case. A slide-level agent analyzes WSIs independently and produces inconsistent estimates, leading to an inaccurate answer. PlanPath formulates an exploration plan, retrieves question-relevant slides, and performs cross-slide tissue-level, cell-level, and morphological analysis. When evidence is insufficient, it retrieves tumor-bearing slides. By integrating evidence across slides and scales, PlanPath estimates approximately 90%, consistent with the reference answer, demonstrating the value of question-driven slide selection, multiscale analysis, and iterative cross-slide evidence integration. Additional Case studies are provided in Appendix D.

Table 3: Prompt token consumption per patient-level question (K tokens/Question ↓).
<table><tr><td>Method</td><td>Gastric</td><td>Renal</td></tr><tr><td>PathAgent</td><td>874.42</td><td>499.50</td></tr><tr><td>PathNavigate</td><td>234.69</td><td>128.90</td></tr><tr><td>GIANT</td><td>1130.88</td><td>579.84</td></tr><tr><td>BEACON</td><td>452.27</td><td>249.01</td></tr><tr><td>PlanPath</td><td>164.32</td><td>120.03</td></tr></table>

Table 4: Ablation study of PlanPath components on two patient-level datasets. Cell Quant. denotes the Cell-level Quantification Tool. Results are reported as Total (%).
<table><tr><td>Configuration</td><td>Renal</td><td>Gastric</td><td>Overall</td></tr><tr><td>PlanPath (Full)</td><td>72.71</td><td>60.24</td><td>65.62</td></tr><tr><td>w/o Slide Router</td><td>71.29</td><td>58.50</td><td>64.02</td></tr><tr><td>w/o Tissue Analysis</td><td>69.43</td><td>56.45</td><td>62.05</td></tr><tr><td>w/o Cell Quant.</td><td>72.57</td><td>59.59</td><td>65.19</td></tr><tr><td>w/o Tissue Analysis &amp; Cell Quant.</td><td>69.57</td><td>57.31</td><td>62.60</td></tr></table>

![](images/8244f266c2f05bba9c825995e37ca6aa67cc7736218ae30d2791a9f6a67acf49.jpg)  
Figure 4: Case study comparing slide-level agents and PlanPath for patient-level VQA.

## 5 RELATED WORKS

Pathology Foundation and Vision–Language Models. Recent pathology foundation models have improved transferable representation learning for histopathology. Vision foundation models such as Virchow (Vorontsov et al., 2024), Prov-GigaPath (Xu et al., 2024), and CHIEF (Wang et al., 2024) learn generalizable morphological representations through large-scale pretraining. Vision–language models such as PLIP (Huang et al., 2023) and CONCH (Lu et al., 2024b) align pathological images with natural language. Approaches, including PathChat (Lu et al., 2024c), TITAN (Ding et al., 2025), SlideChat (Chen et al., 2025b), and WSI-LLaVA (Liang et al., 2025), extend multimodal understanding and visual question answering to whole-slide images. These advances provide strong foundations for pathological representation learning and image–text reasoning, but most still operate on predefined image inputs or individual WSIs and do not explicitly address question-driven evidence acquisition and integration across multiple slides from the same patient.

Pathology Agents for Whole-Slide Image Reasoning. Recent pathology agents have further moved beyond static WSI prediction toward active diagnostic evidence acquisition through ROI localization, magnification adjustment, tool invocation, and iterative reasoning. Representative approaches such as PathAgent (Chen et al., 2025a), GIANT (Buckley et al., 2025), and PathNavigate (Yang et al., 2026) demonstrate the effectiveness of adaptive evidence acquisition for whole-slide image reasoning. However, most existing methods (Ghezloo et al., 2025; Lyu et al., 2025; Ge et al., 2026) remain focused on within-slide exploration and primarily reason from local visual observations. They provide limited support for selecting question-relevant slides from a patient’s WSI collection, dynamically expanding the slide search when evidence is insufficient, and systematically aggregating quantitative evidence across slides. PlanPath extends pathology agents from withinslide navigation to patient-level cross-slide evidence acquisition and integration.

## 6 CONCLUSION

In this work, we formally define patient-level pathology visual question answering and propose PlanPath, a patient-level agentic framework for question-driven slide selection, hierarchical evidence acquisition, quantitative analysis, and cross-slide integration. We also construct two patientlevel VQA datasets for renal and gastric cancer to support systematic evaluation. Experiments demonstrate the effectiveness of PlanPath and its compatibility with existing slide-level pathology agents. We hope this work advances pathology agents from single-slide analysis toward more comprehensive patient-level reasoning.

## AI USE STATEMENT

Generative AI tools were used for language polishing and to assist with dataset construction, including candidate question–answer generation. All AI-assisted content was rigorously reviewed and verified by the authors and domain experts. AI was not used to independently conduct experiments, analyze results, or draw scientific conclusions. The authors take full responsibility for the accuracy, integrity, scientific validity, and final content of this work.

## REFERENCES

Benjamin Adjadj, Pierre-Antoine Bannier, Guillaume Horent, Sebastien Mandela, Gary Klajer, Aurore Lyon, Kathryn Schutte, Ulysse Marteau, Valentin Gaury, Laura Dumont, et al. 2026. Toward comprehensive cellular characterization of H&E slides. Journal ofPathology Informatics (2026), 100696.

Thomas A Buckley, Kian R Weihrauch, Katherine Latham, Andrew Z Zhou, Padmini A Manrai, and Arjun K Manrai. 2025. Navigating gigapixel pathology images with large multimodal models. arXiv preprint arXiv:2511.19652 (2025).

Jingyun Chen, Linghan Cai, Zhikang Wang, Yi Huang, Songhan Jiang, Shenjin Huang, Hongpeng Wang, and Yongbing Zhang. 2025a. Pathagent: Toward interpretable analysis of wholeslide pathology images via large language model-based agentic reasoning. arXiv preprint arXiv:2511.17052 (2025).

Pingyi Chen, Chenglu Zhu, Sunyi Zheng, Honglin Li, and Lin Yang. 2024b. Wsi-vqa: Interpreting whole slide images by generative visual question answering. In European Conference on Computer Vision. Springer, 401–417.

Richard J Chen, Tong Ding, Ming Y Lu, Drew FK Williamson, Guillaume Jaume, Andrew H Song, Bowen Chen, Andrew Zhang, Daniel Shao, Muhammad Shaban, et al. 2024a. Towards a generalpurpose foundation model for computational pathology. Nature medicine 30, 3 (2024), 850–862.

Ying Chen, Guoan Wang, Yuanfeng Ji, Yanjun Li, Jin Ye, Tianbin Li, Ming Hu, Rongshan Yu, Yu Qiao, and Junjun He. 2025b. Slidechat: A large vision-language assistant for whole-slide pathology image understanding. In 2025 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 5134–5143.

Tong Ding, Sophia J Wagner, Andrew H Song, Richard J Chen, Ming Y Lu, Andrew Zhang, Anurag J Vaidya, Guillaume Jaume, Muhammad Shaban, Ahrong Kim, et al. 2025. A multimodal whole-slide foundation model for pathology. Nature Medicine (2025), 1–13.

Jiusong Ge, Yingkang Zhan, Wenjie Zhao, Di Zhang, Ke Wang, Jiashuai Liu, Chunze Yang, Chengzu Li, Jian Zhang, Yuxin Dong, et al. 2026. Thinking in Scales: Accelerating Gigapixel Pathology Image Analysis via Adaptive Continuous Reasoning. arXiv preprint arXiv:2605.19491 (2026).

Fatemeh Ghezloo, Mehmet Saygin Seyfioglu, Rustin Soraki, Wisdom O Ikezogwo, Beibin Li, Tejoram Vivekanandan, Joann G Elmore, Ranjay Krishna, and Linda Shapiro. 2025. Pathfinder: A multi-modal multi-agent system for medical diagnostic decision-making applied to histopathology. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 23431– 23441.

Sunan He, Yuxiang Nie, Hongmei Wang, Shu Yang, Yihui Wang, Zhiyuan Cai, Zhixuan Chen, Yingxue Xu, Linshan Wu, Ngai Shing Cheng, et al. 2026. Towards generalizable AI in medicine via Generalist–Specialist Collaboration. Nature Biomedical Engineering (2026), 1–15.

Wentao Huang, Weimin Lyu, Peiliang Lou, Qingqiao Hu, Xiaoling Hu, Shahira Abousamra, Wenchao Han, Ruifeng Guo, Jiawei Zhou, Chao Chen, et al. 2026. Act Like a Pathologist: Tissue-Aware Whole Slide Image Reasoning. arXiv preprint arXiv:2603.00667 (2026).

Zhi Huang, Federico Bianchi, Mert Yuksekgonul, Thomas J Montine, and James Zou. 2023. A visual–language foundation model for pathology image analysis using medical twitter. Nature medicine 29, 9 (2023), 2307–2316.

Onur C Koyun, Yongxin Guo, Hao Lu, Muhammet Fatih Demir, Abbas Alili, Nabil Rahoui, Diana Cardona, and Metin N Gurcan. 2026. Slide Selection Introduces Sampling-Induced Prediction Uncertainty in Digital Pathology AI: Evidence from Multi-Slide Breast Cancer Cohorts. Cancers 18, 16 (2026), 2540.

Yuci Liang, Xinheng Lyu, Wenting Chen, Meidan Ding, Jipeng Zhang, Xiangjian He, Song Wu, Xiaohan Xing, Sen Yang, Xiyue Wang, et al. 2025. Wsi-llava: A multimodal large language model for whole slide image. In 2025 IEEE/CVF International Conference on Computer Vision (ICCV). IEEE, 22718–22727.

Ming Y Lu, Bowen Chen, Drew FK Williamson, Richard J Chen, Ivy Liang, Tong Ding, Guillaume Jaume, Igor Odintsov, Long Phi Le, Georg Gerber, et al. 2024a. A visual-language foundation model for computational pathology. Nature medicine 30, 3 (2024), 863–874.

Ming Y Lu, Bowen Chen, Drew FK Williamson, Richard J Chen, Ivy Liang, Tong Ding, Guillaume Jaume, Igor Odintsov, Long Phi Le, Georg Gerber, et al. 2024b. A visual-language foundation model for computational pathology. Nature Medicine 30 (2024), 863–874.

Ming Y Lu, Bowen Chen, Drew FK Williamson, Richard J Chen, Melissa Zhao, Aaron K Chow, Kenji Ikemura, Ahrong Kim, Dimitra Pouli, Ankush Patel, et al. 2024c. A multimodal generative AI copilot for human pathology. Nature 634, 8033 (2024), 466–473.

Xinheng Lyu, Yuci Liang, Wenting Chen, Meidan Ding, Jiaqi Yang, Guolin Huang, Daokun Zhang, Xiangjian He, and Linlin Shen. 2025. Wsi-agents: A collaborative multi-agent system for multimodal whole slide image analysis. arXiv preprint arXiv:2507.14680 (2025).

OpenAI. 2026a. GPT-5.5 System Card. OpenAI Deployment Safety Hub. Published April 23, 2026.

OpenAI. 2026b. GPT-5.6 System Card. OpenAI Deployment Safety Hub. Published July 9, 2026.

Qwen Team. 2026. Qwen3.5: Towards Native Multimodal Agents. Qwen Technical Blog.

Andrew Sellergren, Chufan Gao, Fereshteh Mahvar, Timo Kohlberger, Fayaz Jamil, Madeleine Traverse, Alberto Tono, Bashir Sadjad, Lin Yang, Charles Lau, et al. 2026. Medgemma 1.5 technical report. arXiv preprint arXiv:2604.05081 (2026).

Mehmet Saygin Seyfioglu, Wisdom O Ikezogwo, Fatemeh Ghezloo, Ranjay Krishna, and Linda Shapiro. 2024. Quilt-llava: Visual instruction tuning by extracting localized narratives from opensource histopathology videos. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). IEEE, 13183–13192.

Yuxuan Sun, Yixuan Si, Chenglu Zhu, Kai Zhang, Zhongyi Shui, Bowen Ding, Tao Lin, and Lin Yang. 2026. CPathAgent: An Agent-based Foundation Model for Interpretable High-Resolution Pathology Image Analysis Mimicking Pathologists’ Diagnostic Logic. Advances in Neural Infor mation Processing Systems 38 (2026), 101673–101731.

Geert JLH Van Leenders, Theodorus H Van Der Kwast, David J Grignon, Andrew J Evans, Glen Kristiansen, Charlotte F Kweldam, Geert Litjens, Jesse K McKenney, Jonathan Melamed, Nicholas Mottet, et al. 2020. The 2019 International Society of Urological Pathology (ISUP) consensus conference on grading of prostatic carcinoma. The American journal of surgical pathology 44, 8 (2020), e87.

Eugene Vorontsov, Alican Bozkurt, Adam Casson, George Shaikovski, Michal Zelechowski, Kristen Severson, Eric Zimmermann, James Hall, Neil Tenenholtz, Nicolo Fusi, Ellen Yang, Philippe Mathieu, Alexander van Eck, Donghun Lee, Julian Viret, Eric Robert, Yi Kan Wang, Jeremy D. Kunz, Matthew C. H. Lee, Jan H. Bernhard, Ran A. Godrich, Gerard Oakley, Ewan Millar, Matthew Hanna, Hannah Wen, Juan A. Retamero, William A. Moye, Razik Yousfi, Christopher Kanan, David S. Klimstra, Brandon Rothrock, Siqi Liu, and Thomas J. Fuchs. 2024. A foundation model for clinical-grade computational pathology and rare cancers detection. Nature Medicine (2024).

Xiyue Wang, Junhan Zhao, Eliana Marostica, Wei Yuan, Jietian Jin, Jiayu Zhang, Ruijiang Li, Hongping Tang, Kanran Wang, Yu Li, et al. 2024. A pathology foundation model for cancer diagnosis and prognosis prediction. Nature 634, 8035 (2024), 970–978.

Bryan Wong, Xun Xu, Huazhu Fu, Nancy F Chen, and Mun Yong Yi. 2026. Beyond Relevance: Bayesian Evidence Acquisition for Agentic Whole-Slide Image Reasoning. arXiv preprint arXiv:2608.05757 (2026).

Hanwen Xu, Naoto Usuyama, Jaspreet Bagga, Sheng Zhang, Rajesh Rao, Tristan Naumann, Cliff Wong, Zelalem Gero, Javier Gonzalez, Yu Gu, Yanbo Xu, Mu Wei, Wenhui Wang, Shuming Ma,´ Furu Wei, Jianwei Yang, Chunyuan Li, Jianfeng Gao, Jaylen Rosemon, Tucker Bower, Soohee Lee, Roshanthi Weerasinghe, Bill J. Wright, Ari Robicsek, Brian Piening, Carlo Bifulco, Sheng Wang, and Hoifung Poon. 2024. A whole-slide foundation model for digital pathology from real-world data. Nature (2024).

Chunze Yang, Qidong Liu, Wenjie Zhao, Yue Tang, Jiusong Ge, Di Zhang, Jiashuai Liu, Lei Wu, Junbo Lu, Ni Zhang, et al. 2026. PathNavigate: A Training-Free Pathology Agent with Surprise-Guided Scan and Shared Slide Memory for Whole-Slide Image VQA. arXiv preprint arXiv:2605.23559 (2026).

Wenchuan Zhang, Penghao Zhang, Jingru Guo, Tao Cheng, Jie Chen, Shuwan Zhang, Zhang Zhang, Yuhao Yi, and Hong Bu. 2026. Patho-r1: A multimodal reinforcement learning-based pathology expert reasoner. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 40. 28418– 28426.

# PlanPath: Exploration-Guided Orchestration for Patient-Level Pathology Reasoning

## Supplementary Material

## Table of Contents

A Additional Dataset Details 14   
A.1 Patient-Level Pathology VQA Dataset Construction 14   
A.2 Dataset Statistics and Question Distribution 14   
A.3 Ethics Approval and Data Availability 15   
B Implementation Details 16   
B.1 Model Parameter Settings 16   
B.2 Implementation of Slide Router . 17   
B.3 Implementation of Tissue-Level Analysis Tool . 19   
B.4 Baseline Settings 21   
C Additional Experimental Results 21   
C.1 Detailed Results on Patient-Level Benchmarks . 21   
C.2 Comparison of Baseline Aggregation Strategies 21   
C.3 Detailed Compatibility Results 23   
C.4 Detailed Ablation Results . 24   
C.5 Detailed Quantitative Evaluation 25   
D Additional Qualitative Analysis 25   
D.1 Visualization of Quantitative Reasoning Cases 25   
D.2 Visualization of Cross-Slide Patient-Level Reasoning 26

## A ADDITIONAL DATASET DETAILS

## A.1 PATIENT-LEVEL PATHOLOGY VQA DATASET CONSTRUCTION

This section supplements Section 2.1 with details of the clinical source records, patient-level organization, question preparation, and evaluation inputs for PatientVQA-Gastric and PatientVQA-Renal.

Clinical records and patient-level organization. PatientVQA-Gastric draws on pathology records for early and advanced gastric cancer, while PatientVQA-Renal draws on records from patients undergoing partial or radical nephrectomy. The source records provide diagnostic descriptions and structured fields for disease classification and pathological staging. Pathology identifiers link question–answer annotations to patient-level WSI collections. WSIs sharing a pathology identifier are grouped into one collection, and slide filenames are resolved to their stored full paths. All 213 patients in the final datasets match a source record by pathology identifier and have a corresponding WSI collection. Consistent with the inclusion scope in the main text, the reported statistics include only patients with retained questions and corresponding WSIs.

Question generation and verification. GPT-5.6 Sol is used to derive candidate question–answer pairs from the pathology reports, with reference answers grounded in explicitly documented diagnostic findings. The questions cover tumor type, grade, local invasion, margin status, lymph node involvement, and pathological staging. As described in Section 2.1, two pathologists cross-review the candidate pairs, excluding those with medical or logical errors, ambiguous wording, insufficient report support, or inadequate visual evidence in the available WSIs. Each retained question is linked to the patient’s WSI collection and has a reference answer for evaluation.

Question formats and option order. The questions comprise yes/no, single-choice, and openended short-answer questions. All 501 single-choice questions have four candidate options. We permute their order while preserving the question wording, option contents, and correct answer. The reference answer is stored as the correct option text, independent of its position. The resulting option order is saved in the final question files and remains fixed throughout evaluation.

Evaluation samples. Each evaluation sample associates a diagnostic question with the corresponding patient’s WSI collection and, where applicable, its answer options. Depending on the question, the relevant visual evidence may occur within one slide or across multiple slides. Pathology reports are used for question construction and reference answer verification and are not provided to models during inference. Reference answers are reserved for evaluation.

Figure 5 summarizes the dataset construction process. Table 5 shows one example of each question type from both datasets. Within each dataset, the three examples belong to the same patient and are associated with the same WSI collection.

## A.2 DATASET STATISTICS AND QUESTION DISTRIBUTION

Table 6 summarizes the final PatientVQA-Gastric and PatientVQA-Renal datasets. Consistent with Section 2.2, the reported statistics include only patients with retained questions and corresponding WSIs. Together, the datasets contain 213 patients, 3,245 WSIs, and 1,623 question–answer pairs. Figure 6 visualizes the distributions of WSIs and questions per patient in the two datasets.

Patient-level characteristics. The median numbers of WSIs per patient are 21 in Gastric and 8 in Renal, with ranges of 13–58 and 2–36, respectively. The numbers of questions per patient have medians of 9 and 6, with ranges of 5–14 and 2–13. Table 6 also reports the corresponding means. Figure 6(a) shows the distribution of WSI counts per patient in groups of five, while Figure 6(b) shows the distribution of question counts per patient. Numbers above the bars indicate the number of patients in each group.

Question distribution. The datasets contain 595 yes/no, 501 single-choice, and 527 open-ended short-answer questions in total. Their counts and proportions within each dataset are reported in Table 6. Among single-choice questions, the correct answer appears at positions A, B, C, and D in 47, 71, 60, and 81 Gastric questions, respectively, and in 61, 56, 60, and 65 Renal questions, respectively.

![](images/aa6d7ec50c7b0f8c759ef371803e15f0e2ad0b1c23f1640e68a392c2683eb46c.jpg)  
Figure 5: Construction of PatientVQA-Gastric and PatientVQA-Renal. Pathology reports and WSI collections are linked by pathology identifier. GPT-5.6 Sol generates candidate question–answer pairs from the reports. Two pathologists cross-review the candidates for clinical correctness, report support, and answerability from the available WSIs. Each retained question–answer pair is associated with the corresponding patient’s full WSI collection.

Table 5: Examples from the final question files. Each dataset’s three examples belong to one patient. Options are shown in their stored evaluation order. Reference answers are used only for evaluation.
<table><tr><td>Type</td><td>Question and options</td><td>Answer</td></tr><tr><td colspan="3">PatientVQA-Gastric</td></tr><tr><td>Short answer</td><td>What is the deepest level of invasion of the primary tumor in this pa- Submucosa tient?</td><td></td></tr><tr><td>Single-choice</td><td>What is the Lauren classification of the primary tumor in this patient? Mixed type Options: A: Diffuse type; B: Indeterminate type; C: Intestinal type;</td><td></td></tr><tr><td>Yes/No</td><td>D: Mixed type. Is tumor identified at any submitted margin in this patient? Options: Yes; No.</td><td>No</td></tr><tr><td colspan="3">PatientVQA-Renal</td></tr><tr><td>Short answer</td><td>Which pT category is assigned to this patient&#x27;s primary renal tumor? pT1</td><td></td></tr><tr><td>Single-choice</td><td>What is the highest WHO/ISUP grade of the primary tumor in this Grade 2 patient?</td><td></td></tr><tr><td>Yes/No</td><td>Options: A: Grade 4; B: Grade 2; C: Grade 3; D: Grade 1. Does the primary tumor in this patient invade the renal fibrous cap- No sule? Options: Yes; No.</td><td></td></tr></table>

## A.3 ETHICS APPROVAL AND DATA AVAILABILITY

All patient-derived data used in this study were collected and processed with approval from the relevant institutional ethics committees and in accordance with applicable ethical and privacy requirements. The question–answer pairs of PatientVQA-Gastric and PatientVQA-Renal are publicly available at https://github.com/JSGe-AI/PlanPath\_1. The original WSIs are available upon request from the corresponding author.

Table 6: Statistics of PatientVQA-Gastric and PatientVQA-Renal. Question-type percentages are calculated over all questions within the corresponding dataset.
<table><tr><td>Statistic</td><td>Gastric</td><td>Renal</td></tr><tr><td>Patients</td><td>104</td><td>109</td></tr><tr><td>WSIs</td><td>2,227</td><td>1,018</td></tr><tr><td>Question-answer pairs WSIs per patient (mean)</td><td>923</td><td>700 9.34</td></tr><tr><td>WSIs per patient (median) WSIs per patient (range)</td><td>21.41 21 13-58</td><td>8 2-36</td></tr><tr><td>Questions per patient (mean)</td><td>8.88</td><td>6.42</td></tr><tr><td>Questions per patient (median)</td><td>9</td><td>6</td></tr><tr><td>Questions per patient (range)</td><td>5-14</td><td>2-13</td></tr><tr><td>Yes/no</td><td>268 (29.0%)</td><td>327 (46.7%)</td></tr><tr><td>Single-choice</td><td></td><td></td></tr><tr><td>Open-ended short-answer</td><td>259 (28.1%) 396 (42.9%)</td><td>242 (34.6%) 131 (18.7%)</td></tr></table>

(a) WSIs per patient  
![](images/45c06687cc73d2a23ab03ba9500192bc1cfaaa9e48149d80ca8603d7af0161be.jpg)

(b) Questions per patient  
![](images/ad441047b49075b29a1fff0311c855445fc3229a50f2fa6bb891812dbcce9ed5.jpg)  
Figure 6: Patient-level distributions in PatientVQA-Gastric and PatientVQA-Renal. (a) Number of WSIs per patient, grouped into intervals of five. (b) Number of questions per patient. Bar heights and the numbers above nonzero bars indicate patient counts.

## B IMPLEMENTATION DETAILS

## B.1 MODEL PARAMETER SETTINGS

The model configurations and inference parameters of the main PlanPath modules are summarized in Table 7. The Agent Brain is implemented with Qwen3.5-4B, consisting of 32 layers with a hid den size of 2560, and is responsible for exploration planning, evidence verification, and final answer generation. The Morphology Description Tool employs Patho-R1-7B (Qwen2.5-VL) with a context length of 8,192 tokens and single-image input to generate question-independent morphological descriptions. The Slide Router uses TITAN to extract 768-dimensional WSI embeddings and rank all slides of a patient according to text–WSI semantic similarity. For candidate region retrieval, PLIP encodes each ROI with an input size of 224 × 224 and produces 512-dimensional features for semantic relevance ranking.

Table 7: Implementation details of PlanPath.
<table><tr><td>Module</td><td>Model / Architecture</td><td>Key Parameters</td><td>Main Function</td></tr><tr><td>Agent Brain</td><td>Qwen3.5-4B</td><td>32 layers; hidden size 2560</td><td>Exploration planning, evidence verification, and answer generation.</td></tr><tr><td>Morphology Description Tool</td><td>Patho-R1-7B (Qwen2.5-VL)</td><td>8,192-token context; single-image input</td><td>Question-independent morphological description.</td></tr><tr><td>Slide Router</td><td>TITAN</td><td>768-D embeddings</td><td>Text-WSI similarity ranking across patient slides.</td></tr><tr><td>ROI Retrieval</td><td>PLIP</td><td>CLIP-based; 224 × 224 input; 512-D features</td><td>Semantic ranking of candidate ROIs at 5× and above.</td></tr><tr><td>Tissue-level Analysis Tool</td><td>Tissue Feature Prototype Bank</td><td>768-D features; softmax temperature 0.04</td><td>Localization of target tissues, interfaces, and heterogeneous regions.</td></tr><tr><td>Cell-level Quantification Tool</td><td>HistoPLUS</td><td>20×/40× models; 784 × 784 input; confidence thr=0.5</td><td>Cell detection, instance segmentation, classification, and density quantification.</td></tr></table>

Table 8: Scale-specific settings for hierarchical evidence acquisition in PlanPath.
<table><tr><td>Scale</td><td>WSI Level-0 Crop (px)</td><td>Patho-R1 Input (px)</td><td>PLIP Input (px)</td><td>Main Purpose</td><td>HistoPLUS</td></tr><tr><td>5×</td><td> $4 0 9 6 \times 4 0 9 6$ </td><td> $5 1 2 \times 5 1 2$ </td><td> $2 2 4 \times 2 2 4$ </td><td>Tissue architecture, lesion distribution, and tumor interfaces.</td><td>Not used.</td></tr><tr><td>20×</td><td> $1 0 2 4 \times 1 0 2 4$ </td><td> $5 1 2 \times 5 1 2$ </td><td> $2 2 4 \times 2 2 4$ </td><td>Cellular organization, epithelial lesions, and tumor-vessel/nerve relations.</td><td>0.50 µm/px; 784×784 input.</td></tr><tr><td>40×</td><td> $5 1 2 \times 5 1 2$ </td><td> $5 1 2 \times 5 1 2$ </td><td> $2 2 4 \times 2 2 4$ </td><td>Nuclear/cytoplasmic morphology, tumor subtype, mitoses, and subtle foci.</td><td>0.25 µm/px; 784× 784 input.</td></tr></table>

The Tissue-level Analysis Tool operates on a pre-constructed Tissue Feature Prototype Bank with 768-dimensional feature representations. The softmax temperature for prototype matching is set to 0.04, enabling localization of target tissues, tissue interfaces, and morphologically heterogeneous regions. For cell-level analysis, the Cell-level Quantification Tool employs HistoPLUS with dedicated models at 20× and 40× magnifications. Both models use an input size of 784 × 784, with the cell confidence threshold set to 0.50, supporting cell detection, instance segmentation, classification, and density quantification.

To support coarse-to-fine hierarchical evidence acquisition, PlanPath adopts scale-specific WSI cropping and analysis settings at 5×, 20×, and 40×, as summarized in Table 8. At 5×, a 4096×4096 region is cropped from the Level-0 WSI to capture tissue architecture, lesion distribution, and tumor interfaces; HistoPLUS is not activated at this scale. At 20×, the Level-0 crop size is reduced to 1024 × 1024 for examining cellular organization, epithelial lesions, and local tumor–vessel/nerve relationships. HistoPLUS can be invoked at this scale with a resolution of 0.50 µm/px. At 40×, a 512 × 512 Level-0 region is used for detailed inspection of nuclear and cytoplasmic morphology, tumor subtype, mitotic figures, and subtle suspicious foci, with HistoPLUS operating at 0.25 µm/px.

Across all observation scales, the input to Patho-R1 is resized to 512 × 512, while the PLIP input is standardized to 224×224. These scale-specific settings provide a consistent interface across modules while enabling PlanPath to progressively transition from tissue-level structural assessment to local morphological inspection and cell-level quantification during hierarchical evidence acquisition.

## B.2 IMPLEMENTATION OF SLIDE ROUTER

Input, Output, and Retrieval Description Construction. The Slide Router is used only to determine the priority order in which the WSIs of the current patient are examined. Its outputs consist of the ranked slides and their corresponding routing scores, which are not treated as histopathological evidence. The inputs include all WSIs from the same patient and the slide-level retrieval descriptions generated by the Agent Brain for the current question.

For each question, the Agent Brain generates 1–3 concise English caption-style descriptions that characterize the tissue appearances that should be preferentially searched for at the slide level. These descriptions may include dataset-level organ context, but do not contain slide filenames, specimensite labels, or reference answers. For Yes/No questions, the retrieval descriptions cover both the target finding and an assessable contrasting state. For MCQ questions, they cover the major morphological axes associated with different candidate answers, thereby avoiding a retrieval direction dominated by a single option. All retrieval descriptions are used only for slide ranking and do not directly participate in downstream answer generation.

Slide and Text Representations. Each WSI is encoded by the Slide Router into a 768-dimensional global slide representation, $\mathbf { v } _ { i } \in \mathbb { R } ^ { 7 6 8 }$ . In practice, the visual representation of each slide is computed only once and stored in single precision, avoiding repeated WSI encoding for different questions from the same patient. After loading, each representation is checked for dimensional validity, numerical finiteness, and non-zero norm, and is then re-normalized using $L _ { 2 }$ normalization.

The retrieval descriptions are processed by a text encoder aligned with the visual encoder in the same semantic embedding space. The text encoder uses a context window of length 128, with the final position reserved for the global representation token. Inference is performed on GPU in half precision, and the resulting text representations are converted to single precision and renormalized with $L _ { 2 }$ normalization. Let $\mathbf { t } _ { j }$ denote the representation of the j-th retrieval description. The normalized slide and text representations are defined as

$$
\hat { \mathbf { v } } _ { i } = \frac { \mathbf { v } _ { i } } { \Vert \mathbf { v } _ { i } \Vert _ { 2 } } , \qquad \hat { \mathbf { t } } _ { j } = \frac { \mathbf { t } _ { j } } { \Vert \mathbf { t } _ { j } \Vert _ { 2 } } .
$$

Similarity Computation and Patient-level Ranking. The Slide Router performs retrieval and ranking only within the slide collection of the current patient. For slide $S _ { i }$ and retrieval description $d _ { j }$ , the matching score is defined as the cosine similarity between their normalized representations:

$$
\begin{array} { r } { c _ { i j } = \hat { \mathbf v } _ { i } ^ { \top } \hat { \mathbf t } _ { j } . } \end{array}
$$

When multiple retrieval descriptions are generated for a question, the final routing score of slide $S _ { i }$ is defined as

$$
r _ { i } = \operatorname* { m a x } _ { 1 \leq j \leq M } c _ { i j } , \qquad M \in \{ 1 , 2 , 3 \} .
$$

The max aggregation treats the retrieval descriptions as complementary morphological hypotheses, ensuring that a slide strongly matching any one target pattern is not suppressed by low similarity to the remaining descriptions. In addition to the final routing score, the system records the retrieval description producing the maximum score and the similarity scores against all descriptions for auditability.

All slides of the current patient are ranked in descending order according to $r _ { i }$ . When two slides have identical routing scores, the anonymized slide ID is used as a deterministic tie-breaker to ensure reproducible ranking across repeated runs.

Candidate Slide Pool Construction. By default, the top 5 analyzable slides form the initial candidate pool. The pool size is further adjusted according to the evidence requirements of the current question.

For extremum-seeking tasks or focal spatial-interface tasks, the candidate pool is expanded to include at least the top 8 slides. For coverage-sensitive tasks that require sufficient support for patientlevel negative, extremum, or counting conclusions, all analyzable slides are retained. Examples include assessment of the deepest gastric wall invasion and pT stage, perineural or lymphovascular invasion, renal capsular involvement, and regional lymph-node status or pN stage.

Candidate slides are subsequently processed by ROI-level analysis in routing order. For presencedetection questions, exploration may terminate early once direct positive evidence passes the Evidence Gate. In contrast, tasks requiring sufficient candidate-pool coverage continue to inspect the remaining slides until the predefined coverage and stopping criteria are satisfied.

A slide is included in the effective candidate pool only when its global representation, original WSI, and the data required for downstream ROI analysis are all accessible. Slides missing a global representation cannot be assigned a routing score. Slides that can be ranked at the slide level but lack the original image or ROI-analysis data are marked as unanalyzable and excluded from the candidate pool. All missing or exclusion states are recorded in the execution log. If all slides of a patient lack valid global representations, the Slide Router terminates the routing process and returns an error state.

Caching and Reproducibility. To reduce repeated computation across multiple questions from the same patient, the system caches text representations, slide representations, and complete ranking results. The text-cache key is determined by the normalized set of retrieval descriptions. The sliderepresentation cache is bound to the corresponding representation file, while the persistent ranking cache is further conditioned on the current question, Evidence Plan, model state, and slide state. Any change in the relevant inputs triggers recomputation.

The caching mechanism only reuses deterministic intermediate results under identical inputs. It introduces no additional clinical information and does not modify the candidate-pool construction or stopping rules. Regardless of whether a ranking is obtained through real-time computation or cache reuse, the similarity scores, ranking positions, and matched retrieval descriptions produced by the Slide Router are used solely to determine the subsequent slide exploration order. They are not treated as diagnostic evidence and are not directly propagated to final answer generation.

## B.3 IMPLEMENTATION OF TISSUE-LEVEL ANALYSIS TOOL

Module Design. The Tissue-level Analysis Tool is a weakly supervised spatial analysis module based on tissue feature prototypes. Rather than directly diagnosing an ROI, it converts a WSI into a spatial tissue map with soft class probabilities, providing a unified representation for target-region retrieval, tissue composition quantification, and tissue-interface localization. The module consists of three stages: tissue prototype bank construction, patch-level prototype matching, and region-level spatial aggregation.

The prototype bank is constructed once during development. During inference, patch representations are extracted from the current WSI and matched against the fixed prototype bank to generate a soft tissue map. Question-independent patch representations and tissue maps are cached to avoid redundant computation across multiple questions from the same slide.

Tissue Prototype Bank Construction. The gastric prototype bank was constructed from TCGA-STAD, comprising 1,051 WSIs and 1,944,093 pathology patch representations of 768 dimensions. The renal prototype bank was constructed from TCGA-RCC, comprising 934 WSIs and 3,408,240 patch representations. All patch representations were first L<sub>2</sub>-normalized and then clustered using MiniBatch K-means with K = 100. The mini-batch size was set to 8,192, with three initializations and a fixed random seed of 42.

For each cluster, the 30 high-resolution patches closest to the cluster center were manually reviewed to determine the dominant tissue category, morphological purity, and secondary tissue components. After review, 11 tissue categories were retained for TCGA-STAD and 8 for TCGA-RCC. Because hemorrhagic regions in TCGA-STAD did not form an independent dominant cluster, 16 additional patches with a purity of at least 0.90 were selected from mixed clusters to supplement the hemorrhage category.

Representative Prototype Selection. For each reviewed cluster, the 10 patches closest to the cluster center were selected as representative prototypes, while limiting each WSI to at most two selected patches per cluster to reduce slide-specific bias. Randomly sampled and boundary patches were used only for cluster review and were not included in the final prototype bank.

The resulting banks contained 1,016 representative patches for TCGA-STAD and 1,000 for TCGA-RCC. Before inference, representations belonging to the same cluster–tissue-category pair were averaged and re-normalized, yielding 102 inference prototypes for TCGA-STAD and 100 for TCGA-RCC. This design reduces noise from individual patches while preserving intra-class morphological diversity across different clusters. Detailed prototype statistics are summarized in Table 9.

Table 9: Statistics of the tissue feature prototype banks used by the Tissue-level Analysis Tool.
<table><tr><td>Cohort</td><td>#WSIs</td><td>#Patches</td><td>K</td><td>Tissue Category</td><td>Dominant Clusters</td><td>Prototype Patches</td><td>Inference Prototypes</td></tr><tr><td rowspan="10">TCGA-STAD 1,051</td><td rowspan="10"></td><td></td><td>Artifact / background</td><td>15</td><td>150</td><td>15</td></tr><tr><td></td><td>Non-tumor mucosa / epithelium</td><td>3</td><td>30</td><td>3</td></tr><tr><td>Gland-forming tumor</td><td></td><td>21</td><td>210</td><td>21</td></tr><tr><td>Diffuse / poorly cohesive tumor</td><td></td><td>16</td><td>160</td><td>16</td></tr><tr><td></td><td>Desmoplastic stroma / tumor bed</td><td>19</td><td>190</td><td>19</td></tr><tr><td>100</td><td>Immune / lymphoid tissue</td><td>3 1</td><td>30 10</td><td>3</td></tr><tr><td>Necrosis / ulcer debris</td><td></td><td></td><td></td><td>1</td></tr><tr><td>Mucin pool</td><td></td><td></td><td>10</td><td>1</td></tr><tr><td>Hemorrhage / blood†</td><td></td><td></td><td>16</td><td>2</td></tr><tr><td>Adipose / perigastric soft tissue Muscularis</td><td>8</td><td></td><td>80</td><td>8 13</td></tr><tr><td rowspan="10"></td><td rowspan="10"></td><td>Total</td><td>13</td><td></td><td>130 1,016</td><td>102</td></tr><tr><td>Artifact / background</td><td></td><td>100 18</td><td>180</td><td></td></tr><tr><td>Viable renal tumor</td><td></td><td>36</td><td>360</td><td>18 36</td></tr><tr><td>Normal renal parenchyma</td><td></td><td></td><td></td><td>6</td></tr><tr><td>Interstitium / fibrosis</td><td></td><td>6</td><td>60 230</td><td>23</td></tr><tr><td>100 Immune / lymphoid tissue</td><td></td><td>23 2</td><td>20</td><td>2</td></tr><tr><td>Necrosis / debris</td><td></td><td>3</td><td>30</td><td>3</td></tr><tr><td>Hemorrhage / blood</td><td></td><td>8</td><td>80</td><td>8</td></tr><tr><td>Adipose / soft tissue</td><td></td><td>4</td><td>40</td><td>4</td></tr><tr><td>Total</td><td></td><td>100</td><td>1,000</td><td>100</td></tr></table>

Note: Dominant Clusters denotes the number of manually reviewed clusters assigned primarily to each tissue category. Prototype Patches denotes the representative patches retained in the prototype bank, while Inference Prototypes denotes the prototypes obtained by averaging representations within each cluster–category group. <sup>†</sup>The TCGA-STAD hemorrhage category was supplemented with 16 high-purity patches from two mixed source clusters; therefore, it has no independent dominant cluster but yields two inference prototypes.

Patch-level Tissue Matching. For a patch representation $\mathbf { z } _ { i }$ from the target WSI and the k-th normalized prototype $\mathbf { p } _ { c , k }$ of tissue category $c ,$ the category score is defined as the maximum cosine similarity over all prototypes belonging to that category:

$$
s _ { i , c } = \operatorname* { m a x } _ { k } \hat { \mathbf { z } } _ { i } ^ { \top } \hat { \mathbf { p } } _ { c , k } .
$$

This within-class maximum allows different morphological subtypes of the same tissue category to be represented by separate prototypes.

The category scores are converted into soft probabilities using a temperature parameter $\tau = 0 . 0 4 \mathrm { : }$

$$
q _ { i , c } = \frac { \exp { \left[ \left( s _ { i , c } - \operatorname* { m a x } _ { c ^ { \prime } } s _ { i , c ^ { \prime } } \right) / \tau \right] } } { \sum _ { c ^ { \prime } } \exp { \left[ \left( s _ { i , c ^ { \prime } } - \operatorname* { m a x } _ { c ^ { \prime \prime } } s _ { i , c ^ { \prime \prime } } \right) / \tau \right] } } .
$$

A patch is marked as unknown when the highest class probability is below 0.30 or when the raw similarity margin between the highest- and second-highest-scoring classes is below 0.01. For each patch, the system records its spatial coordinates, soft class probabilities, predicted class, confidence, class margin, normalized entropy, and unknown status, thereby constructing a soft tissue map in the original WSI coordinate system.

Regional Quantification and Interface Analysis. For a candidate region R, the Tissue-level Analysis Tool aggregates the soft probabilities of all intersecting patches to estimate tissue composition, target-tissue scores, mean confidence, unknown ratio, and tissue heterogeneity.

For whole-slide quantification, the soft area of tissue category c is defined as

$$
A _ { c } = \sum _ { i \in \mathcal { T } _ { \mathrm { k n o w n } } } a _ { i } q _ { i , c } ,
$$

and its relative tissue fraction is

$$
F _ { c } = \frac { A _ { c } } { \sum _ { c ^ { \prime } } A _ { c ^ { \prime } } } ,
$$

where $a _ { i }$ denotes the area of patch i in the Level-0 coordinate system and $\mathcal { T } _ { \mathrm { k n o w n } }$ denotes patches not labeled as unknown. The proportion of known tissue is additionally used as a quality-control criterion before reporting tissue fractions.

For a tissue pair $( c _ { 1 } , c _ { 2 } )$ , interface candidates are identified by combining their soft-probability cooccurrence with spatial proximity. High-scoring neighboring patches from different tissue categories are localized, and the midpoint of the best-matching pair is used as a candidate interface center for subsequent ROI relocation and higher-magnification inspection. For slide-edge analysis, the tissue map is rasterized and boundary-connected background regions are identified using flood fill, allowing true peripheral regions to be distinguished from internal clefts or artifacts.

The resulting tissue composition, proportion, and interface information is used for ROI ranking, spatial relocation, and explicit tissue-level quantification. Importantly, the predicted tissue classes and probabilities are not treated as direct evidence for tumor subtype, grade, stage, or invasion; such conclusions require confirmation through subsequent morphological observation.

## B.4 BASELINE SETTINGS

For a fair comparison, all baseline methods follow their official implementations and recommended inference settings unless otherwise specified. We use Qwen3.5-4B as the unified model for patientlevel output aggregation across all applicable baselines. For slide-level pathology agents, Qwen3.5- 4B is additionally used for within-slide evidence integration and final slide-level answer generation, ensuring a consistent reasoning backbone across agentic baselines. Each WSI is first processed independently according to the corresponding baseline pipeline, after which the resulting slide-level outputs are aggregated by Qwen3.5-4B into a single patient-level answer. Apart from these unified integration and aggregation settings, all model architectures, visual encoders, preprocessing procedures, tool configurations, and inference parameters follow the official baseline implementations.

## C ADDITIONAL EXPERIMENTAL RESULTS

## C.1 DETAILED RESULTS ON PATIENT-LEVEL BENCHMARKS

The main text reports a compact comparison on PatientVQA-Gastric and PatientVQA-Renal using Total, MCQ, Yes/No, and Open accuracy. Tables 10 and 11 provide the complete per-dataset results and additionally report BLEU-1 and ROUGE-L for open-ended questions.

On PatientVQA-Gastric, PlanPath achieves a BLEU-1 score of 46.66 and a ROUGE-L score of 51.79, exceeding the strongest baselines by 17.98 and 7.47 percentage points, respectively. On PatientVQA-Renal, the corresponding scores reach 54.87 and 59.75, with improvements of 23.48 and 30.38 points. Together with the accuracy results reported in the main text, these results show that the improvement of PlanPath on open-ended questions is consistent across both exact-match and text-overlap metrics.

## C.2 COMPARISON OF BASELINE AGGREGATION STRATEGIES

To examine whether baseline performance depends on how independently generated slide-level predictions are combined, we compare majority voting with Qwen3.5-4B aggregation, which is used for the baseline results in the main text. Both strategies operate on the same saved slide-level predictions, while the underlying visual inference remains unchanged.

Majority Voting. Each non-empty and parseable slide-level answer contributes one vote, and the most frequent answer is selected as the patient-level prediction. For open-ended questions, model specific parsers extract concise answers by removing answer prefixes and truncating at predefined sentence or punctuation boundaries. The extracted answers are normalized for case, whitespace, hyphens, and predefined grade expressions before identical answers are grouped. Empty or unparseable outputs are excluded. If no valid answer remains, the patient-level prediction is set to None and counted as incorrect. Ties are resolved using prediction confidence when available, fol lowed by a deterministic model-specific rule if necessary.

Qwen3.5-4B Aggregation. Qwen3.5-4B receives the question, answer options when applicable, and all valid slide-level answers from the same patient, and selects a single patient-level answer. For open-ended questions, it is instructed to return the selected answer verbatim without paraphrasing, merging, or expanding it. The aggregation model receives no images, pathology reports, or reference answers.

Table 10: Detailed performance on PatientVQA-Gastric. BLEU-1 and ROUGE-L are additionally reported for open-ended questions. All values are reported as percentages. The best and second-best results in each column are shown in bold and underlined, respectively.
<table><tr><td>Method</td><td>Input</td><td>Total</td><td>MCQ</td><td>Yes/No</td><td>Open</td><td>BLEU-1</td><td>ROUGE-L</td></tr><tr><td>Pretrained MLLM Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.5 (OpenAI, 2026a)</td><td>Slide-T</td><td>35.10</td><td>39.77</td><td>30.97</td><td>34.85</td><td>28.22</td><td>36.84</td></tr><tr><td>GPT-5.6-Luna (OpenAI, 2026b)</td><td>Slide-T</td><td>37.70</td><td>36.29</td><td>50.75</td><td>29.80</td><td>18.65</td><td>27.70</td></tr><tr><td>Qwen3.5-4B (Qwen Team, 2026)</td><td>Slide-T</td><td>50.38</td><td>42.86</td><td>65.30</td><td>45.20</td><td>28.68</td><td>44.11</td></tr><tr><td>Qwen3.5-9B (Qwen Team, 2026)</td><td>Slide-T</td><td>47.24</td><td>41.70</td><td>62.31</td><td>40.66</td><td>25.95</td><td>44.32</td></tr><tr><td>Supervised Pathology Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MedDr (He et al., 2026)</td><td>Slide-T</td><td>30.44</td><td>30.50</td><td>64.18</td><td>7.58</td><td>2.85</td><td>5.77</td></tr><tr><td>WSI-VQA (Chen et al., 2024b)</td><td>WSI</td><td>32.83</td><td>44.79</td><td>47.39</td><td>15.15</td><td>14.02</td><td>12.39</td></tr><tr><td>TITAN (Ding et al., 2025)</td><td>WSI</td><td></td><td>26.64</td><td>39.93</td><td></td><td></td><td></td></tr><tr><td>Quilt-LLaVA (Seyfioglu et al., 2024)</td><td>Patch</td><td>29.47</td><td>51.35</td><td>35.82</td><td>10.86</td><td>4.64</td><td>13.27</td></tr><tr><td>WSI-LLaVA (Liang et al., 2025)</td><td>WSI</td><td>32.50</td><td>22.78</td><td>67.91</td><td>14.90</td><td>20.93</td><td>20.49</td></tr><tr><td>MedGemma-1.5-4B (Sellergren et al., 2026)</td><td>Slide-T</td><td>50.81</td><td>52.12</td><td>64.93</td><td>40.40</td><td>10.44</td><td>15.01</td></tr><tr><td>SlideChat (Chen et al., 2025b)</td><td>WSI</td><td>34.78</td><td>26.25</td><td>63.43</td><td>20.96</td><td>10.32</td><td>15.82</td></tr><tr><td>Agentic Systems</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PathAgent (Chen et al., 2025a)</td><td>WSI</td><td>33.91</td><td>40.93</td><td>38.81</td><td>26.01</td><td>13.80</td><td>14.44</td></tr><tr><td>PathNavigate (Yang et al., 2026)</td><td>WSI</td><td>34.45</td><td>43.24</td><td>58.96</td><td>12.12</td><td>8.25</td><td>9.07</td></tr><tr><td>GIANT (Buckley et al., 2025)</td><td>WSI</td><td>37.27</td><td>44.40</td><td>33.21</td><td>35.35</td><td>14.55</td><td>31.33</td></tr><tr><td>BEACON (Wong et al., 2026)</td><td>WSI</td><td></td><td>33.98</td><td>38.81</td><td></td><td></td><td></td></tr><tr><td>PlanPath (ours)</td><td>WSI</td><td>60.24</td><td>57.92</td><td>70.15</td><td>55.05</td><td>46.66</td><td>51.79</td></tr></table>

Table 11: Detailed performance on PatientVQA-Renal. BLEU-1 and ROUGE-L are additionally reported for open-ended questions. All values are reported as percentages. The best and second-best results in each column are shown in bold and underlined, respectively.
<table><tr><td>Method</td><td>Input</td><td>Total</td><td>MCQ</td><td>Yes/No</td><td>Open</td><td>BLEU-1</td><td>ROUGE-L</td></tr><tr><td>Pretrained MLLM Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.5 (OpenAI, 2026a)</td><td>Slide-T</td><td>40.71</td><td>23.97</td><td>60.55</td><td>22.14</td><td>17.86</td><td>19.86</td></tr><tr><td>GPT-5.6-Luna (OpenAI, 2026b)</td><td>Slide-T</td><td>49.29</td><td>47.11</td><td>62.08</td><td>21.37</td><td>24.75</td><td>23.85</td></tr><tr><td>Qwen3.5-4B (Qwen Team, 2026)</td><td>Slide-T</td><td>55.14</td><td>50.00</td><td>69.11</td><td>29.77</td><td>21.67</td><td>20.94</td></tr><tr><td>Qwen3.5-9B (Qwen Team, 2026)</td><td>Slide-T</td><td>55.57</td><td>46.69</td><td>71.56</td><td>32.06</td><td>31.39</td><td>27.19</td></tr><tr><td>Supervised Pathology Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MedDr (He et al., 2026)</td><td>Slide-T</td><td>55.86</td><td>58.26</td><td>70.34</td><td>15.27</td><td>16.76</td><td>9.78</td></tr><tr><td>WSI-VQA (Chen et al., 2024b)</td><td>WSI</td><td>15.86</td><td>13.22</td><td>20.18</td><td>9.92</td><td>1.72</td><td>0.93</td></tr><tr><td>TITAN (Ding et al., 2025)</td><td>WSI</td><td></td><td>13.64</td><td>57.80</td><td></td><td></td><td></td></tr><tr><td>Quilt-LLaVA (Seyfioglu et al., 2024)</td><td>Patch</td><td>37.00</td><td>61.98</td><td>29.97</td><td>8.40</td><td>9.87</td><td>10.15</td></tr><tr><td>WSI-LLaVA (Liang et al., 2025)</td><td>WSI</td><td>49.00</td><td>35.12</td><td>71.87</td><td>17.56</td><td>21.24</td><td>16.09</td></tr><tr><td>MedGemma-1.5-4B (Sellergren et al., 2026)</td><td>Slide-T</td><td>60.57</td><td>59.50</td><td>73.39</td><td>30.53</td><td>25.26</td><td>29.37</td></tr><tr><td>SlideChat (Chen et al., 2025b)</td><td>WSI</td><td>44.43</td><td>17.77</td><td>71.56</td><td>25.95</td><td>9.41</td><td>14.32</td></tr><tr><td>Agentic Systems</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PathAgent (Chen et al., 2025a)</td><td>WSI</td><td>33.14</td><td>46.28</td><td>30.28</td><td>16.03</td><td>23.89</td><td>19.84</td></tr><tr><td>PathNavigate (Yang et al., 2026)</td><td>WSI</td><td>55.14</td><td>57.85</td><td>67.58</td><td>19.08</td><td>21.46</td><td>25.38</td></tr><tr><td>GIANT (Buckley et al., 2025)</td><td>WSI</td><td>47.14</td><td>34.30</td><td>63.30</td><td>30.53</td><td>16.50</td><td>21.11</td></tr><tr><td>BEACON (Wong et al., 2026)</td><td>WSI</td><td></td><td>47.11</td><td>25.08</td><td></td><td></td><td></td></tr><tr><td>PlanPath (ours)</td><td>WSI</td><td>72.71</td><td>78.51</td><td>74.92</td><td>56.49</td><td>54.87</td><td>59.75</td></tr></table>

Results. Tables 12 and 13 report the results using majority voting, while the corresponding Qwen3.5-4B aggregation results are reported in the main-text Table 1. The effect of aggregation varies across baselines, with majority voting improving some methods while reducing others. On PatientVQA-Gastric, the strongest baseline under majority voting achieves 53.30% Total Accuracy, compared with 60.24% for PlanPath, corresponding to a 6.94 percentage-point difference. Under Qwen3.5-4B aggregation, the strongest baseline achieves 50.81%, corresponding to a 9.43-point difference. On PatientVQA-Renal, the strongest baseline reaches 60.43% with majority voting and 60.57% with Qwen3.5-4B aggregation, compared with 72.71% for PlanPath, yielding differences of

Table 12: Performance on PatientVQA-Gastric using majority-voting aggregation for slide-level baselines. PlanPath is shown unchanged for reference; results using Qwen3.5-4B aggregation are reported in the main-text Table 1. Best and second-best results are shown in bold and underlined, respectively.
<table><tr><td>Method</td><td>Input</td><td>Total</td><td>MCQ</td><td>Yes/No</td><td>Open</td><td>BLEU-1</td><td>ROUGE-L</td></tr><tr><td>Pretrained MLLM Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.5 (OpenAI, 2026a)</td><td>Slide-T</td><td>45.50</td><td>45.17</td><td>46.27</td><td>45.20</td><td>24.32</td><td>22.05</td></tr><tr><td>GPT-5.6-Luna (OpenAI, 2026b)</td><td>Slide-T</td><td>43.45</td><td>41.31</td><td>64.55</td><td>30.56</td><td>21.19</td><td>26.45</td></tr><tr><td>Qwen3.5-4B (Qwen Team, 2026)</td><td>Slide-T</td><td>53.30</td><td>47.49</td><td>64.18</td><td>49.75</td><td>34.64</td><td>50.09</td></tr><tr><td>Qwen3.5-9B (Qwen Team, 2026)</td><td>Slide-T</td><td>51.25</td><td>46.33</td><td>64.18</td><td>45.71</td><td>35.43</td><td>51.52</td></tr><tr><td>Supervised Pathology Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MedDr (He et al., 2026)</td><td>Slide-T</td><td>34.99</td><td>35.91</td><td>64.18</td><td>14.65</td><td>5.97</td><td>12.79</td></tr><tr><td>WSI-VQA (Chen et al., 2024b)</td><td>WSI</td><td>30.01</td><td>37.45</td><td>49.63</td><td>11.87</td><td>13.93</td><td>12.32</td></tr><tr><td>TITAN (Ding et al., 2025)</td><td>WSI</td><td></td><td>33.20</td><td>45.90</td><td></td><td></td><td></td></tr><tr><td>Quilt-LLaVA (Seyfioglu et al., 2024)</td><td>Patch</td><td>29.47</td><td>51.35</td><td>35.82</td><td>10.86</td><td>4.96</td><td>13.04</td></tr><tr><td>WSI-LLaVA (Liang et al., 2025)</td><td>WSI</td><td>32.18</td><td>22.78</td><td>69.03</td><td>13.38</td><td>23.08</td><td>20.49</td></tr><tr><td>MedGemma-1.5-4B (Sellergren et al., 2026)</td><td>Slide-T</td><td>44.85</td><td>51.74</td><td>64.18</td><td>27.27</td><td>12.46</td><td>15.01</td></tr><tr><td>SlideChat (Chen et al., 2025b)</td><td>WSI</td><td>35.32</td><td>27.80</td><td>64.18</td><td>20.71</td><td>8.37</td><td>15.45</td></tr><tr><td>Agentic Systems</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PathAgent (Chen et al., 2025a)</td><td>WSI</td><td>40.85</td><td>41.70</td><td>51.87</td><td>32.83</td><td>17.70</td><td>16.02</td></tr><tr><td>PathNavigate (Yang et al., 2026)</td><td>WSI</td><td>42.04</td><td>47.88</td><td>79.48</td><td>12.88</td><td>2.96</td><td>3.40</td></tr><tr><td>GIANT (Buckley et al., 2025)</td><td>WSI</td><td>47.45</td><td>47.49</td><td>60.07</td><td>38.89</td><td>25.52</td><td>38.81</td></tr><tr><td>BEACON (Wong et al., 2026)</td><td>WSI</td><td></td><td>36.68</td><td>46.27</td><td></td><td></td><td></td></tr><tr><td>PlanPath (ours)</td><td>WSI</td><td>60.24</td><td>57.92</td><td>70.15</td><td>55.05</td><td>46.66</td><td>51.79</td></tr></table>

Table 13: Performance on PatientVQA-Renal using majority-voting aggregation for slide-level baselines. PlanPath is shown unchanged for reference; results using Qwen3.5-4B aggregation are reported in the main-text Table 1. Best and second-best results are shown in bold and underlined, respectively.
<table><tr><td>Method</td><td>Input</td><td>Total</td><td>MCQ</td><td>Yes/No</td><td>Open</td><td>BLEU-1</td><td>ROUGE-L</td></tr><tr><td>Pretrained MLLM Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.5 (OpenAI, 2026a)</td><td>Slide-T</td><td>46.29</td><td>35.95</td><td>62.69</td><td>24.43</td><td>22.12</td><td>21.69</td></tr><tr><td>GPT-5.6-Luna (OpenAI, 2026b)</td><td>Slide-T</td><td>55.00</td><td>62.81</td><td>63.30</td><td>19.85</td><td>24.54</td><td>21.62</td></tr><tr><td>Qwen3.5-4B (Qwen Team, 2026)</td><td>Slide-T</td><td>55.57</td><td>48.76</td><td>70.64</td><td>30.53</td><td>24.77</td><td>21.87</td></tr><tr><td>Qwen3.5-9B (Qwen Team, 2026)</td><td>Slide-T</td><td>54.29</td><td>41.74</td><td>71.87</td><td>33.59</td><td>32.71</td><td>28.72</td></tr><tr><td>Supervised Pathology Models</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MedDr (He et al., 2026)</td><td>Slide-T</td><td>56.00</td><td>59.50</td><td>70.03</td><td>14.50</td><td>15.37</td><td>8.98</td></tr><tr><td>WSI-VQA (Chen et al., 2024b)</td><td>WSI</td><td>18.14</td><td>13.64</td><td>25.08</td><td>9.16</td><td>0.87</td><td>0.56</td></tr><tr><td>TITAN (Ding et al., 2025)</td><td>WSI</td><td></td><td>16.12</td><td>63.61</td><td></td><td></td><td></td></tr><tr><td>Quilt-LLaVA (Seyfioglu et al., 2024)</td><td>Patch</td><td>36.71</td><td>61.98</td><td>29.97</td><td>6.87</td><td>9.62</td><td>8.66</td></tr><tr><td>WSI-LLaVA (Liang et al., 2025)</td><td>WSI</td><td>55.86</td><td>50.41</td><td>75.23</td><td>17.56</td><td>20.35</td><td>13.80</td></tr><tr><td>MedGemma-1.5-4B (Sellergren et al., 2026)</td><td>Slide-T</td><td>60.00</td><td>61.16</td><td>71.56</td><td>29.01</td><td>22.67</td><td>27.02</td></tr><tr><td>SlideChat (Chen et al., 2025b)</td><td>WSI</td><td>44.00</td><td>18.18</td><td>70.64</td><td>25.19</td><td>9.68</td><td>13.04</td></tr><tr><td>Agentic Systems</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>PathAgent (Chen et al., 2025a)</td><td>WSI</td><td>37.86</td><td>51.65</td><td>36.09</td><td>16.79</td><td>19.42</td><td>16.63</td></tr><tr><td>PathNavigate (Yang et al., 2026)</td><td>WSI</td><td>60.43</td><td>59.09</td><td>77.06</td><td>21.37</td><td>23.79</td><td>26.47</td></tr><tr><td>GIANT (Buckley et al., 2025)</td><td>WSI</td><td>51.14</td><td>40.08</td><td>67.58</td><td>30.53</td><td>17.41</td><td>19.94</td></tr><tr><td>BEACON (Wong et al., 2026)</td><td>WSI</td><td></td><td>51.65</td><td>22.63</td><td></td><td></td><td></td></tr><tr><td>PlanPath (ours)</td><td>WSI</td><td>72.71</td><td>78.51</td><td>74.92</td><td>56.49</td><td>54.87</td><td>59.75</td></tr></table>

12.28 and 12.14 percentage points, respectively. These results show that the performance advantage of PlanPath is maintained under both baseline aggregation strategies.

## C.3 DETAILED COMPATIBILITY RESULTS

Figure 7 further reports the per-question-type compatibility results of PlanPath with PathAgent, PathNavigate, GIANT, and BEACON. For PathAgent, PathNavigate, and GIANT, PlanPath consistently improves MCQ performance on both datasets, with gains of 2.70–5.79 percentage points on PatientVQA-Gastric and 1.65–7.02 points on PatientVQA-Renal. Larger gains are observed on Yes/No questions, reaching 17.16–27.24 points on Gastric and 1.84–7.95 points on Renal.

![](images/dde008c8252d14c00dec6847c13969c776007eecbd28d364a06804dcf9e71255.jpg)  
Figure 7: Detailed compatibility results of PlanPath with different slide-level pathology agents on PatientVQA-Gastric and PatientVQA-Renal. For PathAgent, PathNavigate, and GIANT, results are reported separately for MCQ, Yes/No, and Open questions. BEACON is evaluated on MCQ and Yes/No questions. Each pair of bars compares the original agent (Base) with the same agent integrated into PlanPath (+PlanPath), and the red annotations indicate the absolute performance change in percentage points.

Table 14: Detailed ablation results of PlanPath on PatientVQA-Gastric and PatientVQA-Renal. Cell Quant. denotes the Cell-level Quantification Tool. All values are reported as percentages.
<table><tr><td rowspan="2">Configuration</td><td colspan="4">PatientVQA-Gastric</td><td colspan="4">PatientVQA-Renal</td></tr><tr><td>Total</td><td>MCQ</td><td>Yes/No</td><td>Open</td><td>Total</td><td>MCQ</td><td>Yes/No</td><td>Open</td></tr><tr><td>PlanPath (Full)</td><td>60.24</td><td>57.92</td><td>70.15</td><td>55.05</td><td>72.71</td><td>78.51</td><td>74.92</td><td>56.49</td></tr><tr><td>w/o Slide Router</td><td>58.50</td><td>57.53</td><td>70.15</td><td>51.26</td><td>71.29</td><td>76.03</td><td>73.70</td><td>56.49</td></tr><tr><td>w/o Tissue Analysis</td><td>56.45</td><td>58.69</td><td>64.55</td><td>49.49</td><td>69.43</td><td>73.97</td><td>72.78</td><td>52.67</td></tr><tr><td>w/o Cell Quant.</td><td>59.59</td><td>58.30</td><td>70.15</td><td>53.28</td><td>72.57</td><td>78.51</td><td>74.92</td><td>55.73</td></tr><tr><td>w/o Tissue Analysis &amp; Cell Quant.</td><td>57.31</td><td>56.37</td><td>64.18</td><td>53.28</td><td>69.57</td><td>73.55</td><td>72.78</td><td>54.20</td></tr></table>

For open-ended questions, the effect is more method-dependent. PlanPath improves PathAgent by 5.30 and 3.05 points on Gastric and Renal, respectively, and improves PathNavigate by 1.52 and 3.06 points. In contrast, GIANT shows small decreases of 2.27 and 1.52 points on the two datasets, indicating that the benefit of patient-level orchestration can vary with the underlying agent for openended generation.

BEACON is evaluated separately on the question types for which outputs are available. Integrating PlanPath improves its MCQ accuracy by 10.42 points on Gastric and 4.96 points on Renal, while Yes/No accuracy increases by 17.91 and 8.25 points, respectively. Overall, these detailed results show that the improvements observed in overall accuracy are supported by consistent gains on structured question types across different underlying agents, while open-ended performance remains more dependent on the generation behavior of the base agent.

## C.4 DETAILED ABLATION RESULTS

Table 14 reports the per-dataset and per-question-type ablation results corresponding to the overall results in the main text. Removing the Tissue-level Analysis Tool causes the largest and most consistent degradation, reducing Total Accuracy by 3.79 and 3.28 percentage points on PatientVQA-Gastric and PatientVQA-Renal, respectively. The Slide Router also contributes to both datasets, with particularly clear effects on Gastric open-ended questions and Renal MCQ performance. Removing the Cell-level Quantification Tool produces a smaller overall decrease, with most of its effect concentrated on open-ended questions. These results further show that the contributions of individual components are task-dependent and not strictly additive across question types.

![](images/4e3e3cf180ba6e0f504be28c97372eed25c08ac7d1b756f966d251d168da467a.jpg)  
Figure 8: Detailed comparison of all baselines on patient-level quantitative questions. Performance is evaluated using Exact Match and tolerance-based accuracy with thresholds of ±5, ±10, ±15, and ±20. Numerical values denote accuracy (%). Color intensity is normalized independently within each evaluation criterion, with darker purple indicating relatively better performance within the same row. Red boxes indicate the best result for each criterion.

## C.5 DETAILED QUANTITATIVE EVALUATION

In the main text, we report the quantitative-question results of representative slide-level pathology agents, including PathAgent, PathNavigate, and GIANT, for a compact comparison with PlanPath. Here, we provide a more comprehensive evaluation against all baselines. Figure 8 summarizes the Exact Match accuracy and tolerance-based accuracy under ±5, ±10, ±15, and ±20 thresholds.

As shown in Figure 8, PlanPath consistently achieves the best performance across all five evaluation criteria. Specifically, PlanPath obtains an Exact Match accuracy of 38.89%, compared with 22.22% for the strongest baselines under this strict criterion. When progressively relaxing the tolerance threshold, its accuracy increases to 61.11%, 72.22%, 72.22%, and 83.33% under ±5, ±10, ±15, and ±20, respectively. The corresponding best baseline results are 22.22%, 44.44%, 44.44%, and 66.67%. These results show that the advantage of PlanPath is not limited to exact numerical matching, but remains consistent across different levels of numerical tolerance, supporting its ability to acquire and integrate quantitative pathological evidence at the patient level.

## D ADDITIONAL QUALITATIVE ANALYSIS

We provide five additional examples from PatientVQA-Gastric and PatientVQA-Renal to comple ment the case study in Fig. 4. These examples illustrate how PlanPath translates a patient-level question into an exploration plan, acquires relevant evidence across slides and magnifications, and integrates the resulting observations into a final answer. The visualizations highlight different evidence requirements for quantitative estimation, cross-slide assessment, and question-specific lesion identification.

## D.1 VISUALIZATION OF QUANTITATIVE REASONING CASES

Figures 9 and 10 present two examples of histological component estimation. Answering these questions requires identifying the target morphology and assessing its extent within the tumor. PlanPath therefore combines tissue-level spatial context, ROI-level morphological observations, and cellular measurements to support area-based aggregation across the selected slides.

Signet-ring component. Figure 9 addresses a question targeting poorly cohesive cells with intracytoplasmic mucin vacuoles and displaced nuclei. Some 20× fields show cohesive glandular or trabecular structures without definite signet-ring morphology, prompting further exploration. Subsequent 40× observations provide supporting cellular detail, which is integrated with tissue-level information to obtain an estimate of approximately 10%. This case illustrates how phenotype-specific retrieval and higher-magnification inspection support estimation of a component with limited extent.

Hepatoid component. In Fig. 10, PlanPath searches for trabecular or solid arrangements of polygonal cells. Higher-magnification observations reveal granular eosinophilic cytoplasm and prominent nucleoli in representative regions. Evidence acquired from five candidate slides supports an estimated hepatoid proportion of approximately 30%. Together, these cases show how componentspecific morphology and tissue-level extent contribute complementary information to patient-level quantitative reasoning.

## D.2 VISUALIZATION OF CROSS-SLIDE PATIENT-LEVEL REASONING

Figures 11, 12, and 13 illustrate cross-slide reasoning for tumor focality, margin status, and additional lesion identification. These tasks differ in both the evidence to be acquired and the conditions under which exploration can terminate. PlanPath maintains the association between each observation and its source tissue while integrating complementary findings across the patient’s slides.

Primary tumor foci. Figure 11 illustrates the use of spatial relationships in focus counting. Plan-Path first localizes separate tissue components and then examines their cellular content at higher magnification. After reviewing five candidate slides, it identifies two independently supported foci. Aggregation preserves the distinction between spatially separate foci and repeated appearances across sections: the final count uses the maximum independently confirmed count within a section, without summing serial-section observations.

Renal resection margin. In Fig. 12, the exploration targets the relationship between epithelial cells and the submitted resection edge. An inked-edge observation shows no epithelial crossing, but this local finding does not immediately end the search. PlanPath continues until all four candidate slides have been examined and integrates the relevant edge and interface observations, returning no carcinoma identified. The example illustrates the broader evidence requirement for a negative margin assessment.

Additional simple cyst. In Fig. 13, the first examined slide does not establish the required cystic architecture. Exploration then moves to another candidate, where 20× observations reveal a single cuboidal epithelial lining with uniform nuclei and no stratification or significant atypia. These findings support a simple cyst, allowing PlanPath to stop after examining two of the five candidates. The decisive evidence is supplied by tissue architecture and lining-cell morphology.

![](images/98e533bf40040ab3038749770c6b24f5ee588c3c91337e414536e1a29439f1ec.jpg)  
Figure 9: Patient-level estimation of the signet-ring component. The exploration plan targets poorly cohesive cells with intracellular mucin and displaced nuclei. Higher-magnification observations provide phenotype-specific support, which is integrated with tissue-level quantitative evidence to estimate a proportion of approximately 10%.

![](images/b07b0c921c484c8886113ac0673cbf8af4b176603e3496888187e7ec7a638b29.jpg)  
Figure 10: Patient-level estimation of the hepatoid component. PlanPath examines trabecular and solid tumor regions and acquires higher-magnification evidence of polygonal cells with eosinophilic cytoplasm. Quantitative integration across the selected slides yields an estimated proportion of approximately 30%.

![](images/057ec680e64164154a77e9baff1033a99f5592b92e98ab57a43017eadbacf077.jpg)  
Figure 11: Cross-slide assessment of primary tumor focality. Tissue-level localization identifies spatially separate candidate components, and higher-magnification observations confirm their cellular content. Integration yields two independently supported foci while avoiding repeated counting across serial sections.

![](images/22d42932879c6f6b0dc942ea3d752051089e6ec06a1f8f24068cb34308122af3.jpg)  
Figure 12: Cross-slide assessment of the renal resection margin. PlanPath localizes relevant cutedge relationships and continues exploration after an initial negative finding to satisfy the four-slide candidate coverage requirement. The integrated answer is no carcinoma identified at the margin.

![](images/8604a8a542e640582fd361798a10dcb8b726b8f423a903e22881f637db96c1c9.jpg)  
Figure 13: Identification of an additional simple cyst. Targeted exploration reveals cystic architecture and a single bland cuboidal lining, providing sufficient positive evidence after two candidate slides. Cellular views provide morphological context.