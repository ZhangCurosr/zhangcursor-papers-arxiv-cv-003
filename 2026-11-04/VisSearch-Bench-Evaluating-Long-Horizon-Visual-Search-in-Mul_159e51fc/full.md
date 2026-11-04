# VisSearch-Bench: Evaluating Long Horizon Visual Search in Multimodal Agents

ZhengXian Wu<sup>1,2,</sup>∗<sup>,</sup>‡, Qianhao Nie<sup>1,2,</sup>∗, Zhuohong Chen<sup>1,</sup>∗, Junjie Gao<sup>2</sup>, Yuyao Zhang<sup>2,3</sup>, Yunyao Yu<sup>1</sup>, Hangrui Xu<sup>1</sup>, Cheng Cheng<sup>2</sup>, Guiyu Ma<sup>2</sup>, Jin Zhang<sup>2</sup>, Xingzhong Xu<sup>2</sup>, Kai Yang<sup>2,</sup>†, Haoqian Wang<sup>1,</sup>†

<sup>1</sup>Tsinghua University <sup>2</sup>Ant Group <sup>3</sup>Renmin University of China ∗Equal contribution, †Corresponding author, ‡Work done during an internship at Ant Group

## Abstract

Multimodal agents use web and image tools to answer questions beyond their internal knowledge. Some search tasks use images mainly to identify an initial entity, then rely on text retrieval. These tasks ofer limited insight into how agents connect information across images, objects, and search steps. We therefore focus on visual context management. This involves organizing and using visual information and linking it to other findings during search. Visually intensive tasks require agents to combine object appearance, local details, or spatial relations with external information. We introduce VisSearch-Bench, with 204 humanwritten tasks across 7 main domains and 25 subdomains. It covers Set-wise Visual Search, Sequential Visual Search, Hybrid Wide–Deep Search, Competitive Visual Disambiguation, and Visual-Structure-Guided Search. Each task provides a reference answer, traceable evidence, and scoring rules. Accuracy measures complete task success, while Process Score measures the correctness and completeness of individual results and subquestion answers. Across 16 multimodal models, the best system fully solves only 26.47% of tasks, even with search tools. These results highlight the challenge of linking visual evidence with search results to produce complete answers.

Website: Project page a Code: GitHub repository Hugging Face: Dataset

Tsinghua University

![](images/f909fea90c43f567caa36c9e345c9b47be352f8873450258854a5cee3297f327.jpg)

![](images/8305539faa6e324f6b3ac57bfb381ba58be7114659613f1a60484524ab4cde94.jpg)  
Figure 1 Accuracy of models and agents on VisSearch-Bench, comparing direct answering with ReAct and DeerFlow. The dashed line marks the average accuracy across the systems shown.

## 1 Introduction

Multimodal large language models (MLLMs) are evolving from one-shot visual question answering systems into agents that can plan, use tools, and interact with the open web over multiple rounds (Luo et al., 2026; Wu et al., 2026b). With text search, image search, webpage browsing, and region-level image inspection, these systems can gather evidence beyond the initial prompt and solve complex information-seeking tasks (Chng et al., 2025; Zhang et al., 2026b). This paradigm is closer to realworld information seeking than visual question answering based only on parametric knowledge.

At the same time, multimodal benchmarks are placing stronger demands on visual information. Models must not only understand images in the prompt, but also combine visual and textual evidence found during search (Hou et al., 2026). MMSearch-Plus (Tao et al., 2026) emphasizes the extraction and propagation of fine-grained visual cues. MM-BrowseComp (Li et al., 2025) and VisBrowse-Bench (Zhang et al., 2026c) require agents to actively obtain visual evidence while browsing. Together, these benchmarks make visual evidence an active component of search.

However, existing multimodal search benchmarks often focus on whether visual evidence is used at some point in a task. How images are used can lead to diferent task demands. In some existing tasks, images mainly serve to identify an initial entity (Wang et al., 2023; Chen et al., 2026). Once the agent obtains the name of a person, object, or place, it can rely mainly on text search. Such tasks provide limited insight into how agents handle multiple images or connect visual information across search steps. More recent benchmarks involve more image searches and tool calls. However, higher counts alone do not imply more complex visual context. We also need to examine how these visual operations are connected and how they support later searches and the final answer.

Real-world information seeking is more than a series of separate image recognition tasks (Zhang et al., 2026b). On the open web, agents need to update their understanding as they gather new evidence (Yao et al., 2022). Early evidence may suggest an initial hypothesis. Later evidence may help compare candidates, rule some out, or identify new ones. New information may also change how an earlier image is understood. An agent therefore needs to connect visual evidence across search steps and use it to guide further search (Hu et al., 2023). For example, finding all works in a catalog that meet a visual condition requires several steps. The agent must first establish the full set of entries, then inspect their images and combine the results. This raises a basic question: Can agents reliably organize and use visual evidence as it accumulates during open-web search?

We use visual context management to describe how agents organize and use visual information and its connections to findings during search. We study this ability through tasks that involve multiple images, multiple objects, and linked search steps. We examine whether agents can connect visual observations with external information to compare objects, filter candidates, and follow clues. We also examine whether they can combine results across steps into a complete and consistent answer.

To evaluate this ability, we introduce VisSearch-Bench, with five task categories that reflect diferent demands of long-horizon visual search. Set-wise Visual Search requires agents to collect and check visual information for a set of objects, then filter, compare, or combine the results. Sequential Visual Search uses visual clues from earlier steps to guide later searches. Hybrid Wide–Deep Search combines step-by-step search with search over a set of objects. Results from one stage determine the search scope or targets of another. Competitive Visual Disambiguation focuses on using fine visual details to distinguish similar-looking candidates. Visual-Structure-Guided Search combines object order, spatial positions, or connections in an image with information found through external search. Agents use this information to locate objects, check conditions, and combine results.

![](images/654bfa34d2ed64bba528a1e85c8ae5309daffff1e4e42dbdf69269783baf8e2f.jpg)  
Figure 2 Overview of VisSearch-Bench. The center presents five task families that require long-horizon visual search, while the surrounding panels show selected examples from the dataset across diverse content domains.

Together, these tasks examine how agents handle multiple images, connect search results, and reason across steps. Figure 2 presents the five categories and representative tasks from diferent domains.

VisSearch-Bench contains 204 human-written tasks across 7 main domains and 25 subdomains. Dataset construction is led by human annotators and supported by automated checks. Annotators build each task from public, searchable sources. They gather the required images, entity relationships, and facts, then write the question and reference answer. They also document the evidence supporting the solution. Automated checks and human review examine source accessibility, answer consistency, and possible shortcuts that bypass the intended visual requirements. For evaluation, we use Accuracy to measure whether the full answer meets the task requirements. We also use a taskspecific Process Score to assess the correctness and completeness of individual results and answers to subquestions. Our main contributions are as follows:

1. We introduce a task-based perspective for evaluating visual context management in open-web search. It focuses on how agents organize information from multiple images, connect diferent objects, and use visual findings to guide later searches.

2. We construct VisSearch-Bench, with 204 human-written tasks across 7 main domains and 25 subdomains. Its five categories cover Set-wise Visual Search, Sequential Visual Search, Hybrid Wide–Deep Search, Competitive Visual Disambiguation, and Visual-Structure-Guided Search.

3. We develop a human-led construction process supported by automated checks. Each task includes evidence sources, a reference answer, and scoring rules. Our evaluation combines fullanswer accuracy with scores for individual results and subquestions. This measures overall task completion while also capturing which parts of an answer are correct or missing.

## 2 Related Work

## 2.1 Vision-Language Models

Vision-language models (VLMs) combine visual information with language understanding (Wu et al., 2026c; Yu et al., 2026). Their capabilities have expanded from single-image recognition and question answering to document reading, multi-image understanding, video analysis, and visual reasoning. InternVL3 (Zhu et al., 2025) learns language and visual capabilities through joint pre-training and supports longer multimodal contexts. Kimi-VL (Du et al., 2025) focuses on highresolution images, multi-image understanding, and long-context reasoning. Qwen3-VL (Bai et al., 2025) supports interleaved text, image, and video inputs, with improvements in multimodal reasoning and spatial understanding. These advances provide a foundation for handling more varied and complex visual inputs.

<table><tr><td>Benchmarks</td><td>Open- Web</td><td>Image Search</td><td>Multi- image</td><td>Multi- hop</td><td>Set Answers</td><td>Evidence Links</td><td>Partial Credit</td></tr><tr><td>BrowseComp</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MMSearch</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BrowseComp-VL</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MM-BrowseComp</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MMSearch-Plus</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VisBrowse-Bench</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BrowseComp-V3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>InterLV-Search</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VDR-Bench</td><td></td><td>1</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>VisSearch-Bench (Ours)</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 1 Comparison of representative open-web and multimodal search benchmarks. ✓ Full; Partial; — Not covered. InterLV-Search refers to its open-web Level 3.

Recent models further combine visual understanding with general reasoning, tool use, and multistep task execution. Qwen3.5 (Qwen Team, 2026a) uses unified vision-language training and reinforcement learning to improve reasoning and agent capabilities. Later models, including Qwen3.6- 35B-A3B (Qwen Team, 2026c) and Qwen3.8-27B (Qwen Team, 2026d), support image and video understanding. They also improve code generation and multi-step task execution. Visual understanding is thus becoming an important part of general-purpose agent models. It supports tasks such as interpreting charts, reading documents, and using information from user interfaces.

Other model families follow a similar direction. GLM-5.3-Flash (GLM-5-Team et al., 2026) introduces native multimodal capabilities to the GLM-5 series. It supports browser and computer use based on visual information. DeepSeek-V4.1-Flash (Xu et al., 2026) supports both visual inputs and long-context reasoning. Kimi K3 (Team et al., 2026a) also provides native image and video understanding. Its one-million-token context window supports extended reasoning and tool interactions.

However, longer context windows do not guarantee efective visual-state management during search (Wu et al., 2026b). In open-web search, agents also need to correctly match objects across images and connect visual observations with external facts. They must use earlier findings to determine later search targets. VisSearch-Bench therefore focuses on visual context management in tasks involving multiple images, multiple objects, and linked search steps. It evaluates whether agents can use these connections to verify objects, compare candidates, filter results, and combine findings across steps.

## 2.2 Agentic Information Seeking

Existing benchmarks evaluate agentic information seeking through reasoning, tool use, and information gathering. GAIA focuses on multi-step reasoning and tool use in realistic tasks (Mialon et al., 2023). BrowseComp requires agents to persistently search the open web for facts that are dificult to find but easy to verify (Wei et al., 2025). WideSearch extends the focus to large-scale information gathering, emphasizing complete results that can be checked item by item (Wong et al., 2025). DeepWideSearch combines multi-hop retrieval with broad information collection to evaluate whether agents can handle both search depth and breadth (Lan et al., 2025).

Recent benchmarks place greater emphasis on answer coverage and information integration. DeepSearchQA (Gupta et al., 2026) requires agents to follow linked search steps, gather information from diferent sources, and produce complete answer lists. GISA supports several answer formats, including individual items, sets, lists, and tables. It also provides human search trajectories to support process analysis and model training (Zhu et al., 2026). DeepResearch Bench II (Li et al., 2026) extends evaluation to long-form research reports. It uses detailed scoring criteria derived from expert reports to assess information coverage, analysis, and presentation.

Research on search methods also examines how agents organize evidence and adjust later search actions. SearchOS (Zhang et al., 2026b) maintains shared records of pending tasks, evidence, information coverage, and failed attempts. These records help coordinate multiple agents, fill information gaps, and reduce repeated unsuccessful searches. Further work uses search trajectories for supervised fine-tuning to connect retrieval decisions more closely with the reasoning process (Chen et al., 2026).

## 2.3 Multimodal Search and Browsing

BrowseComp (Wei et al., 2025) mainly evaluates text-based information seeking on the open web and provides a reference for assessing browsing capabilities. MMSearch (Jiang et al., 2024) in troduces image-text queries and evaluates query rewriting, result reranking, information summarization, and end-to-end answer generation. BrowseComp-VL (Geng et al., 2025) combines visual recognition with complex web retrieval, requiring agents to conduct multi-step searches based on entities in images. Later benchmarks place greater emphasis on the role of visual evidence during search. MMSearch-Plus (Tao et al., 2026) requires agents to extract fine-grained visual clues and verify related facts and sources through multiple rounds of image-text retrieval. VDR-Bench (Zeng et al., 2026) focuses on reducing shortcuts based on textual clues and whole-image matching. It uses retrieval from image regions, entity verification, and multi-hop question construction to place greater demands on visual search. VisBrowse-Bench (Zhang et al., 2026c) emphasizes actively gathering visual information from webpages and evaluates spatial localization, attribute recognition, and reasoning across images.

Other work further expands search structures and process evaluation. InterLV-Search (Hou et al., 2026) examines how visual and textual evidence guide later search steps. Its open-web tasks include single-chain and multi-branch structures that involve comparing multiple objects or search paths. MM-BrowseComp (Li et al., 2025) introduces multimodal evidence in both the input and the browsing process. It uses reasoning checklists to assess the completion of key steps. BrowseComp-V3 (Zhang et al., 2026a) covers alignment within image regions, integration across regions, and reasoning across images. It requires publicly searchable evidence and evaluates both final-answer accuracy and subgoal completion.

![](images/8ae3265233d1cc1cbdec52419927bc7506d15deb887d56f3d130728d07c33d62.jpg)  
Figure 3 Overview of the VisSearch-Bench construction pipeline. Human annotators curate public sources and organize them into traceable evidence graphs; structural constraints instantiate the five task families; each task is paired with a locked answer and gold trajectory; deterministic integrity gates, LLM-based post-filtering, and final human review verify the retained items.

VisSearch-Bench further focuses on visual context management in search involving multiple images, multiple objects, and linked steps. Its tasks involve checking objects individually, combining results across a set, following visual clues, and distinguishing similar candidates. They also include using image structure together with external information to determine search targets. Table 1 compares the main search requirements covered by VisSearch-Bench and existing benchmarks based on their task designs.

## 3 VisSearch-Bench

## 3.1 Task Definition and Taxonomy

We study search in open-web environments where visual context plays a central role. Each task provides a natural-language question q, an optional initial image set $\mathcal { T } _ { 0 }$ that is empty for text-initiated tasks, and a target output format σ. A multimodal agent interacts with the web through tools for text search, image search, reverse image search, webpage access, and image inspection. At step t, the agent selects its next action $a _ { t + 1 }$ based on the interaction history

$$
h _ { t } = \left( q , \mathcal { T } _ { 0 } , \left( a _ { 1 } , o _ { 1 } \right) , \ldots , \left( a _ { t } , o _ { t } \right) \right)\tag{1}
$$

where $O _ { t }$ denotes the textual or visual observation returned by a tool. When the interaction ends, the agent produces a predicted answer $\hat { y }$ in the required format σ.

In VisSearch-Bench, we define visual context as the visual information involved in a task and its connections to object identities, external facts, and search targets. This information may come from input images or from images and webpages found during search. It includes object appearance and lo cal features, as well as the order, positions, and connections of objects. Visual context management refers to organizing, connecting, and using this information to solve a task. This includes checking objects, comparing candidates, filtering results, and combining findings across steps. Based on these requirements, we organize VisSearch-Bench into five task categories. Each category emphasizes a main search requirement.

![](images/0c0d5a78a3b65ff7d674bbf1c05d72284cd37733c5fb6eaa4942c4c035ab1e8a.jpg)  
Figure 4 Domain taxonomy and composition of VisSearch-Bench. The inner ring shows seven primary domains, and the outer ring shows 25 secondary domains. Arc area is proportional to the number of tasks in the 204-question benchmark. The right panel expands the full taxonomy and reports the task count within each category.

1) Set-wise Visual Search. These tasks involve a clearly defined set of objects. The set may come from a catalog specified in the question or consist of objects shown in the input images. For each object, the agent gathers and checks visual information. It then combines this information with relevant facts to filter, compare, or summarize the results. The main challenge is to cover all required objects and correctly link the information for each one. Individual judgments must also remain consistent with the final summary.

2) Sequential Visual Search. These tasks involve multiple search steps that depend on one another. An object, visual clue, or fact found in one step guides later searches. For example, an agent may first identify an artwork in an image, then find records about its artist, collection, or exhibitions. Diferent steps may involve image understanding or text search. The focus is on correctly connecting the information found across steps to produce the required answer.

3) Hybrid Wide–Deep Search. These tasks combine sequential search with search over a set of objects. Earlier findings can define the candidate set to be checked. Filtering this set can then determine the target of the next stage. For example, an agent may start from one work and identify its award and year. It then examines images of works from the same award cycle and selects those that meet the given conditions. Further searches gather information about the selected works. The main challenge is to correctly connect the search scope, individual judgments, and later targets across stages.

4) Competitive Visual Disambiguation. These tasks focus on visual targets that are easily confused with other objects. Candidates may look similar overall but difer in local shapes, textures, markings, or individual parts. The agent compares the input image with retrieved images, descriptions, and source records to verify the target’s identity. These tasks assess the accuracy of visual detail matching and identity verification.

5) Visual-Structure-Guided Search. These tasks use object order, spatial positions, or connections in an image to determine what to search for, in what order, or where to search next. The agent combines this structural information with external search results to locate objects, check conditions, and summarize findings. The main challenge is to correctly connect image structure, object identities, and search results, then use these connections to complete the task.

## 3.2 Dataset Construction

Figure 3 summarizes our human-led construction process with program-assisted verification. The process follows three principles. Visual Indispensability means that critical visual evidence cannot be replaced by the question text, webpage titles, captions, OCR, or model knowledge. Visual Evidence Integration means that tasks require agents to connect visual observations with object identities and external information. Agents use these connections to guide searches, compare candidates, filter objects, or combine results across steps. Public Solvability and Auditability means that each task uses publicly retrievable evidence and has verifiable critical states and a clear reference answer.

Annotators first select public sources that provide a bounded entity set, reliable entityimage links, and independently verifiable factual fields. We fix these sources before

<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Total questions</td><td>204 5</td></tr><tr><td>Number of task families Number of primary domains</td><td>7</td></tr><tr><td>Set-wise Visual Search Reference-solution steps (avg. / max.)</td><td>68 (33.3%) 23.9/51</td></tr><tr><td>Sequential Visual Search Reference-solution steps (avg. / max.)</td><td>34 (16.7%) 14.4 / 19</td></tr><tr><td>Hybrid Wide-Deep Search Reference-solution steps (avg. / max.)</td><td>34 (16.7%) 23.8 / 61</td></tr><tr><td>Competitive Visual Disambiguation Reference-solution steps (avg. / max.)</td><td>34 (16.7%)</td></tr><tr><td>Visual-Structure-Guided Search</td><td>3.9/5 34 (16.7%)</td></tr><tr><td>Reference-solution steps (avg. / max.)</td><td>15.8 / 19</td></tr><tr><td>Tasks with initial images</td><td>187 (91.7%)</td></tr><tr><td>Tasks without initial images</td><td>17 (8.3%)</td></tr><tr><td>Total reference-solution steps Reference-solution steps per task (median / mean)</td><td>3,593 17.0/17.6</td></tr></table>

Table 2 Dataset composition and structural complexity.

question writing. The reference answer therefore does not depend on model search results or subjective judgments. Annotators then organize the entities, images, factual fields, and candidate sets into a traceable evidence structure. They apply the structural constraints of one task family from Section 3.1 and write a question that hides the intermediate nodes and final answer. Automated checks verify source accessibility, field completeness, set coverage, answer recomputation, and possible shortcuts. We discard tasks that bypass critical visual evidence or allow a shorter unintended path. Human reviewers inspect every retained task.

## 3.3 Evaluation Protocol and Metrics

Our primary metric is Accuracy (Acc.). For task i, let $y _ { i } , \hat { y } _ { i }$ , and $\sigma _ { i }$ denote the reference answer, the predicted answer, and the required answer format. We define

$$
a _ { i } = \mathbb { I } [ \mathcal { E } _ { \sigma _ { i } } ( \hat { y } _ { i } , y _ { i } ) = 1 ] ,\tag{2}
$$

$$
\operatorname { A c c } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } a _ { i } .\tag{3}
$$

where $\mathcal { E } _ { \sigma _ { i } }$ compares parsed answers under predefined, task-specific rules. It normalizes entity aliases.   
For unordered sets, it removes duplicates and ignores order. For ordered sequences, it preserves order.

<table><tr><td rowspan=1 colspan=17>Model / Backbone                              Accuracy (Acc., %)                               Process Score (%)Avg.Cult.Soc.Geo.Life Sci.Tech.Bus.Avg.Cult.Soc.Geo.Life Sci.Tech.Bus.</td></tr><tr><td rowspan=1 colspan=17>Open-source Models</td></tr><tr><td rowspan=1 colspan=1>Qwen3.6-35B-A3B</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=2>0.000.00</td><td rowspan=1 colspan=1>5.56</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=2>0.000.00</td><td rowspan=1 colspan=1>9.55</td><td rowspan=1 colspan=2>4.1910.50</td><td rowspan=1 colspan=1>9.94</td><td rowspan=1 colspan=1>17.48</td><td rowspan=1 colspan=2>12.176.46</td><td rowspan=1 colspan=1>2.92</td></tr><tr><td rowspan=1 colspan=1>Qwen3.6-27B</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=2>0.000.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=2>0.000.00</td><td rowspan=1 colspan=1>5.53</td><td rowspan=1 colspan=2>3.229.79</td><td rowspan=1 colspan=1>6.02</td><td rowspan=1 colspan=1>8.46</td><td rowspan=1 colspan=2>4.15 2.29</td><td rowspan=1 colspan=1>7.02</td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-397B-A17B</td><td rowspan=1 colspan=1>1.96</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=2>0.000.00</td><td rowspan=1 colspan=1>11.11</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=2>0.000.00</td><td rowspan=1 colspan=1>12.72</td><td rowspan=1 colspan=2>7.0013.75</td><td rowspan=1 colspan=1>11.05</td><td rowspan=1 colspan=3>21.7011.9313.46</td><td rowspan=1 colspan=1>10.78</td></tr><tr><td rowspan=1 colspan=1>GLM-4.6V</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=5>0.000.000.002.780.00</td><td rowspan=1 colspan=2>0.000.00</td><td rowspan=1 colspan=1>9.24</td><td rowspan=1 colspan=2>4.1410.53</td><td rowspan=1 colspan=1>8.93</td><td rowspan=1 colspan=1>15.82</td><td rowspan=1 colspan=2>7.1412.26</td><td rowspan=1 colspan=1>8.24</td></tr><tr><td rowspan=1 colspan=1>MiniMax-M3</td><td rowspan=1 colspan=1>2.45</td><td rowspan=1 colspan=5>2.080.004.178.330.00</td><td rowspan=1 colspan=2>0.000.00</td><td rowspan=1 colspan=1>9.71</td><td rowspan=1 colspan=2>6.69 9.46</td><td rowspan=1 colspan=1>5.59</td><td rowspan=1 colspan=1>18.03</td><td rowspan=1 colspan=2>9.2110.98</td><td rowspan=1 colspan=1>4.46</td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-235B-A22B-Thinking</td><td rowspan=1 colspan=1>1.47</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>2.78</td><td rowspan=1 colspan=1>2.78</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>11.37</td><td rowspan=1 colspan=2>6.6014.41</td><td rowspan=1 colspan=1>10.13</td><td rowspan=1 colspan=1>16.28</td><td rowspan=1 colspan=2>14.0011.64</td><td rowspan=1 colspan=1>3.61</td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-8B-Instruct</td><td rowspan=1 colspan=1>0.49</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>2.78</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>5.83</td><td rowspan=1 colspan=2>1.7413.62</td><td rowspan=1 colspan=1>1.82</td><td rowspan=1 colspan=1>13.84</td><td rowspan=1 colspan=1>1.58</td><td rowspan=1 colspan=1>6.13</td><td rowspan=1 colspan=1>2.78</td></tr><tr><td rowspan=1 colspan=1>Qwen3-VL-4B-Instruct</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>5.17</td><td rowspan=1 colspan=2>0.5410.09</td><td rowspan=1 colspan=1>4.23</td><td rowspan=1 colspan=1>9.60</td><td rowspan=1 colspan=1>4.57</td><td rowspan=1 colspan=1>5.95</td><td rowspan=1 colspan=1>2.78</td></tr><tr><td rowspan=1 colspan=1>GLM-5.3-Flash</td><td rowspan=1 colspan=1>3.92</td><td rowspan=1 colspan=1>6.25</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>8.66</td><td rowspan=1 colspan=1>8.75</td><td rowspan=1 colspan=1>2.50</td><td rowspan=1 colspan=1>8.13</td><td rowspan=1 colspan=1>17.18</td><td rowspan=1 colspan=1>5.96</td><td rowspan=1 colspan=1>9.03</td><td rowspan=1 colspan=1>3.51</td></tr><tr><td rowspan=1 colspan=1>Qwen3.8-Flash-Next</td><td rowspan=1 colspan=1>3.43</td><td rowspan=1 colspan=1>6.25</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>5.56</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>11.82</td><td rowspan=1 colspan=1>8.06</td><td rowspan=1 colspan=1>8.36</td><td rowspan=1 colspan=1>16.02</td><td rowspan=1 colspan=1>19.06</td><td rowspan=1 colspan=1>9.07</td><td rowspan=1 colspan=1>11.91</td><td rowspan=1 colspan=1>11.75</td></tr><tr><td rowspan=1 colspan=1>Qwen3.8-27B</td><td rowspan=1 colspan=1>2.45</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>11.11</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>6.35</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>3.06</td><td rowspan=1 colspan=1>8.46</td><td rowspan=1 colspan=1>14.82</td><td rowspan=1 colspan=1>6.72</td><td rowspan=1 colspan=1>5.11</td><td rowspan=1 colspan=1>8.70</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-V4.1-Flash</td><td rowspan=1 colspan=1>1.96</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>5.56</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>12.88</td><td rowspan=1 colspan=1>6.93</td><td rowspan=1 colspan=1>13.42</td><td rowspan=1 colspan=1>8.22</td><td rowspan=1 colspan=1>19.33</td><td rowspan=1 colspan=1>13.07</td><td rowspan=1 colspan=1>19.00</td><td rowspan=1 colspan=1>12.84</td></tr><tr><td rowspan=1 colspan=1>Kimi-K3</td><td rowspan=1 colspan=1>5.88</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>11.11</td><td rowspan=1 colspan=1>2.78</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>10.24</td><td rowspan=1 colspan=1>7.73</td><td rowspan=1 colspan=1>9.12</td><td rowspan=1 colspan=1>8.07</td><td rowspan=1 colspan=1>19.26</td><td rowspan=1 colspan=1>9.31</td><td rowspan=1 colspan=1>11.58</td><td rowspan=1 colspan=1>0.00</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=7>Closed-source Models</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=4></td></tr><tr><td rowspan=2 colspan=1>Gemini 3.5 FlashClaude Opus 4.8</td><td rowspan=1 colspan=1>6.37</td><td rowspan=1 colspan=1>6.25</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>2.78</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>22.28</td><td rowspan=1 colspan=1>17.42</td><td rowspan=1 colspan=1>23.11</td><td rowspan=1 colspan=1>29.11</td><td rowspan=1 colspan=1>27.59</td><td rowspan=1 colspan=1>19.29</td><td rowspan=1 colspan=1>24.39</td><td rowspan=1 colspan=1>15.22</td></tr><tr><td rowspan=1 colspan=1>4.90</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>5.56</td><td rowspan=1 colspan=1>5.56</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>16.13</td><td rowspan=1 colspan=1>12.40</td><td rowspan=1 colspan=1>17.51</td><td rowspan=1 colspan=1>11.12</td><td rowspan=1 colspan=1>18.27</td><td rowspan=1 colspan=1>18.21</td><td rowspan=1 colspan=1>17.51</td><td rowspan=1 colspan=1>22.97</td></tr><tr><td rowspan=1 colspan=1>GPT-5.5</td><td rowspan=1 colspan=1>6.37</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>11.11</td><td rowspan=1 colspan=1>5.56</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>22.61</td><td rowspan=1 colspan=1>19.19</td><td rowspan=1 colspan=1>18.83</td><td rowspan=1 colspan=1>30.08</td><td rowspan=1 colspan=1>25.33</td><td rowspan=1 colspan=1>18.46</td><td rowspan=1 colspan=1>32.95</td><td rowspan=1 colspan=1>12.53</td></tr><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=5>Open-source A</td><td rowspan=1 colspan=2>gents</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=3></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>ReAct</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Qwen3.6-35B-A3B</td><td rowspan=1 colspan=1>7.35</td><td rowspan=1 colspan=1>4.17</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>5.56</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>16.6</td><td rowspan=1 colspan=1>0.007</td><td rowspan=1 colspan=1>23.25</td><td rowspan=1 colspan=1>14.44</td><td rowspan=1 colspan=1>22.74</td><td rowspan=1 colspan=1>27.27</td><td rowspan=1 colspan=1>25.56</td><td rowspan=1 colspan=1>24.91</td><td rowspan=1 colspan=1>29.63</td><td rowspan=1 colspan=1>26.77</td></tr><tr><td rowspan=1 colspan=1>GPT-5.5</td><td rowspan=1 colspan=1>25.49</td><td rowspan=1 colspan=1>20.83</td><td rowspan=1 colspan=1>20.83</td><td rowspan=1 colspan=1>33.33</td><td rowspan=1 colspan=1>33.33</td><td rowspan=1 colspan=1>25.00</td><td rowspan=1 colspan=1>29.17</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>49.52</td><td rowspan=1 colspan=1>46.70</td><td rowspan=1 colspan=1>43.42</td><td rowspan=1 colspan=1>60.37</td><td rowspan=1 colspan=1>52.76</td><td rowspan=1 colspan=1>48.02</td><td rowspan=1 colspan=1>59.05</td><td rowspan=1 colspan=1>27.02</td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-397B-A17B</td><td rowspan=1 colspan=1>15.69</td><td rowspan=1 colspan=1>16.67</td><td rowspan=1 colspan=1>16.67</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>16.67</td><td rowspan=1 colspan=1>16.67</td><td rowspan=1 colspan=1>25.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>32.43</td><td rowspan=1 colspan=2>26.1430.82</td><td rowspan=1 colspan=2>26.6733.47</td><td rowspan=1 colspan=1>42.68</td><td rowspan=1 colspan=1>41.19</td><td rowspan=1 colspan=1>20.89</td></tr><tr><td rowspan=1 colspan=1>GLM-5.3-Flash</td><td rowspan=1 colspan=1>20.10</td><td rowspan=1 colspan=3>20.8316.6720.83</td><td rowspan=1 colspan=2>36.1113.89</td><td rowspan=1 colspan=1>16.67</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>22.29</td><td rowspan=1 colspan=7>21.7019.5820.8336.1119.0422.920.00</td></tr><tr><td rowspan=1 colspan=1>Kimi-K3</td><td rowspan=1 colspan=1>26.47</td><td rowspan=1 colspan=3>25.0033.3333.33</td><td rowspan=1 colspan=1>36.11</td><td rowspan=1 colspan=1>22.22</td><td rowspan=1 colspan=1>20.83</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>38.57</td><td rowspan=1 colspan=2>39.5045.00</td><td rowspan=1 colspan=3>40.0042.7534.58</td><td rowspan=1 colspan=2>37.0621.60</td></tr><tr><td rowspan=1 colspan=1>DeepSeek-V4.1-Flash</td><td rowspan=1 colspan=1>22.55</td><td rowspan=1 colspan=1>14.58</td><td rowspan=1 colspan=1>20.83</td><td rowspan=1 colspan=1>29.17</td><td rowspan=1 colspan=1>38.89</td><td rowspan=1 colspan=1>19.44</td><td rowspan=1 colspan=1>25.00</td><td rowspan=1 colspan=1>0.00</td><td rowspan=1 colspan=1>39.47</td><td rowspan=1 colspan=2>38.0333.13</td><td rowspan=1 colspan=2>38.4450.12</td><td rowspan=1 colspan=1>37.50</td><td rowspan=1 colspan=2>44.8323.18</td></tr><tr><td rowspan=1 colspan=1>Qwen3.8-27B</td><td rowspan=1 colspan=1>21.08</td><td rowspan=1 colspan=1>16.67</td><td rowspan=1 colspan=1>25.00</td><td rowspan=1 colspan=1>20.83</td><td rowspan=1 colspan=1>36.11</td><td rowspan=1 colspan=1>13.89</td><td rowspan=1 colspan=1>20.83</td><td rowspan=1 colspan=1>8.33</td><td rowspan=1 colspan=1>33.23</td><td rowspan=1 colspan=2>26.2834.25</td><td rowspan=1 colspan=2>32.1241.48</td><td rowspan=1 colspan=1>32.70</td><td rowspan=1 colspan=2>37.0530.35</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>De</td><td rowspan=1 colspan=1>erFlo</td><td rowspan=1 colspan=1>w</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1>Qwen3.6-35B-A3B</td><td rowspan=1 colspan=1>10.78</td><td rowspan=1 colspan=3>10.42 12.504.17</td><td rowspan=1 colspan=2>11.11 13.89</td><td rowspan=1 colspan=2>16.670.00</td><td rowspan=1 colspan=1>24.26</td><td rowspan=1 colspan=4>17.8926.4517.2129.03</td><td rowspan=1 colspan=1>33.152</td><td rowspan=1 colspan=2>7.7711.49</td></tr><tr><td rowspan=1 colspan=1>GPT-5.5</td><td rowspan=1 colspan=1>20.10</td><td rowspan=1 colspan=5>16.6729.1712.5027.7816.67</td><td rowspan=1 colspan=2>25.008.33</td><td rowspan=1 colspan=1>38.30</td><td rowspan=1 colspan=4>35.34 43.8635.3934.78</td><td rowspan=1 colspan=1>35.33</td><td rowspan=1 colspan=2>51.4138.13</td></tr><tr><td rowspan=1 colspan=1>Qwen3.5-397B-A17B</td><td rowspan=1 colspan=1>10.29</td><td rowspan=1 colspan=5>8.3312.500.0019.4411.11</td><td rowspan=1 colspan=2>12.500.00</td><td rowspan=1 colspan=1>28.37</td><td rowspan=1 colspan=7>19.5432.4422.5933.1340.8829.44 13.11</td></tr></table>

Table 3 Main results on VisSearch-Bench. Avg. denotes the micro-average over all benchmark tasks. Cult., Soc., Geo., Life, Sci., Tech., and Bus. denote Culture, Society, Geography, Life Sciences, Physical Sciences, Technology, and Business, respectively.

## A task is correct only if the prediction contains all and only the required content.

Final-answer accuracy cannot show whether an error comes from the wrong search space, a missed visual candidate, or incorrect evidence aggregation (Wu et al., 2026a). We therefore also report Process Score (PS). It measures the correctness and completeness of individual results and answers to subquestions, using the scoring rules defined for each task. Each task i has $m _ { i }$ labeled critical states, which all systems report before the final answer. Let $[ m _ { i } ] = \{ 1 , \dots , m _ { i } \}$ index these states. We define the set of correctly recovered states as

$$
\mathcal { R } _ { i } = \left\{ j \in [ m _ { i } ] ~ \vert ~ \hat { g } _ { i j } \equiv _ { \tau _ { i j } } g _ { i j } \right\} ,\tag{4}
$$

$$
\mathrm { P S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { \lvert \mathcal { R } _ { i } \rvert } { m _ { i } } .\tag{5}
$$

Here, $\hat { g } _ { i j } \equiv _ { \tau _ { i j } } \ g _ { i j }$ means that predicted state $j$ matches its reference under the corresponding rule. States are aligned by field labels, and diferent names for the same entity are treated as equal. Each match counts as one correct state, so $| \mathcal { R } _ { i } | / m _ { i }$ is the fraction of required states recovered for task i.

<table><tr><td rowspan="2">Model / Backbone</td><td colspan="2">Avg. (204)</td><td colspan="2">Set-wise (68)</td><td colspan="2">Sequential Visual Search Visual Search (34)</td><td colspan="2">Hybrid Wide-Deep Search (34)</td><td colspan="2">Competitive Visual Disambiguation (34)</td><td colspan="2">Visual-Structure- Guided Search (34)</td></tr><tr><td>Process Acc.</td><td>Score</td><td>Acc.</td><td>Process Score</td><td>Acc.</td><td>Process Score</td><td>Acc.</td><td>Process Score</td><td>Acc.</td><td>Process Score</td><td>Acc.</td><td>Process Score</td></tr><tr><td colspan="10">Open-source Models</td><td></td></tr><tr><td>Qwen3.6-35B-A3B</td><td>0.98</td><td>9.55</td><td>0.00</td><td>10.79</td><td>0.00</td><td>6.40 0.00</td><td>5.42</td><td>0.00</td><td></td><td>0.00</td><td>5.88</td><td>23.94</td></tr><tr><td>Qwen3.6-27B</td><td>0.00</td><td>5.53</td><td>0.00</td><td>7.94</td><td>0.00</td><td>5.44</td><td>0.00</td><td>1.53</td><td>0.00</td><td>0.00</td><td>0.00</td><td>10.31</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>1.96</td><td>12.72</td><td>0.00</td><td>13.77</td><td>0.00</td><td>10.48</td><td>0.00</td><td>11.12</td><td>0.00</td><td>0.00</td><td>11.76</td><td>27.17</td></tr><tr><td>GLM-4.6V</td><td>0.49</td><td>9.24</td><td>0.00</td><td>6.94</td><td>0.00</td><td>5.29</td><td>0.00</td><td>10.74</td><td>0.00</td><td>0.00</td><td>2.94</td><td>25.53</td></tr><tr><td>MiniMax-M3</td><td>2.45</td><td>9.71</td><td>0.00</td><td>7.18</td><td>0.00</td><td>7.38</td><td>2.94</td><td>4.89</td><td>0.00</td><td>0.00</td><td>11.76</td><td>31.61</td></tr><tr><td>Qwen3-VL-235B-A22B-Thinking</td><td>1.47</td><td>11.37</td><td>0.00</td><td>13.21</td><td>0.00</td><td>7.28</td><td>0.00</td><td>7.22</td><td>0.00</td><td>0.00</td><td>8.82</td><td>27.27</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>0.49</td><td>5.83</td><td>0.00</td><td>7.15</td><td>0.00</td><td>2.85</td><td>0.00</td><td>4.07</td><td>0.00</td><td>0.00</td><td>2.94</td><td>13.77</td></tr><tr><td>Qwen3-VL-4B-Instruct</td><td>0.00</td><td>5.17</td><td>0.00</td><td>6.69</td><td>0.00</td><td>2.62</td><td>0.00</td><td>3.35</td><td>0.00</td><td>0.00</td><td>0.00</td><td>11.69</td></tr><tr><td>GLM-5.3-Flash</td><td>3.92</td><td>8.66</td><td>0.00</td><td>4.22</td><td>0.00</td><td>4.26</td><td>8.82</td><td>8.82</td><td>0.00</td><td>0.00</td><td>14.71</td><td>30.45</td></tr><tr><td>Qwen3.8-Flash-Next</td><td>3.43</td><td>11.82</td><td>0.00</td><td>6.24</td><td>0.00</td><td>9.56</td><td>8.82</td><td>14.94</td><td>0.00</td><td>0.00</td><td>11.76</td><td>33.96</td></tr><tr><td>Qwen3.8-27B</td><td>2.45</td><td>6.35</td><td>0.00</td><td>4.33</td><td>0.00</td><td>1.03</td><td>0.00</td><td>0.41</td><td>0.00</td><td>0.00</td><td>14.71</td><td>28.02</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>1.96</td><td>12.88</td><td>0.00</td><td>11.69</td><td>0.00</td><td>11.54</td><td>2.94</td><td>13.72</td><td>0.00</td><td>0.00</td><td>8.82</td><td>28.66</td></tr><tr><td>Kimi-K3</td><td>5.88</td><td>10.24</td><td>2.94</td><td>6.69</td><td>2.94</td><td>11.47</td><td>8.82</td><td>10.82</td><td>0.00</td><td>0.00</td><td>17.65</td><td>25.79</td></tr><tr><td colspan="10">Closed-source Models</td><td></td><td></td></tr><tr><td>Gemini 3.5 Flash</td><td>6.37</td><td>22.28</td><td>0.00</td><td>21.40</td><td>2.94</td><td>22.3511.76</td><td></td><td>28.08</td><td>0.00</td><td>0.00</td><td>23.53</td><td>40.44</td></tr><tr><td>Claude Opus 4.8</td><td>4.90</td><td>16.13</td><td>0.00</td><td>12.28</td><td>5.88</td><td>15.14</td><td>11.76</td><td>20.02</td><td>0.00</td><td>0.00</td><td>11.76</td><td>37.09</td></tr><tr><td>GPT-5.5</td><td>6.37</td><td>22.61</td><td>1.47</td><td>21.23</td><td>2.94</td><td>21.36</td><td>8.82</td><td>28.36</td><td>0.00</td><td>0.00</td><td>23.53</td><td>43.47</td></tr><tr><td colspan="10">Open-source Agents</td></tr><tr><td></td><td></td><td></td><td>2.94</td><td>19.81</td><td>ReAct</td><td></td><td></td><td>23.40</td><td>2.94</td><td>2.94</td><td>26.47</td><td>57.05</td></tr><tr><td>Qwen3.6-35B-A3B GPT-5.5</td><td>7.35 25.49</td><td>23.25 49.52</td><td>8.82</td><td>44.10</td><td>2.94 23.53</td><td>16.47 51.73</td><td>5.88 23.53</td><td>48.36</td><td>38.24</td><td>38.24</td><td>50.00</td><td>70.61</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>15.69</td><td>32.43</td><td>0.00</td><td>20.14</td><td></td><td></td><td>14.71</td><td>27.21</td><td>32.35</td><td>29.41</td><td>35.29</td><td>56.05</td></tr><tr><td>GLM-5.3-Flash</td><td></td><td>22.29</td><td>10.29</td><td></td><td>11.7641.60</td><td></td><td></td><td></td><td>35.29</td><td>35.29</td><td>47.06</td><td>51.16</td></tr><tr><td></td><td>20.10</td><td></td><td></td><td>11.58</td><td>8.82</td><td>15.29</td><td>8.82</td><td>8.82</td><td></td><td></td><td></td><td></td></tr><tr><td>Kimi-K3</td><td>26.47</td><td>38.57</td><td>11.76</td><td>24.00</td><td>32.35</td><td>56.34</td><td>14.71</td><td>27.74</td><td>41.18</td><td>44.12</td><td>47.06</td><td>55.22</td></tr><tr><td>DeepSeek-V4.1-Flash Qwen3.8-27B</td><td>22.55 21.08</td><td>39.47</td><td>5.88</td><td>28.28</td><td>26.47</td><td>52.52</td><td>11.76</td><td>20.97</td><td>44.12</td><td>44.12</td><td>41.18</td><td>62.63</td></tr><tr><td></td><td>33.23</td><td></td><td>7.35</td><td>18.78</td><td>23.53 46.91</td><td></td><td>8.82</td><td>20.17</td><td>44.12</td><td>44.12</td><td>35.29</td><td>50.61</td></tr><tr><td colspan="10">DeerFlow</td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.6-35B-A3B</td><td>10.78</td><td>24.26</td><td>1.47</td><td>19.08</td><td>8.82</td><td>20.29</td><td>5.88</td><td>17.55</td><td>35.29</td><td>35.29</td><td>11.76</td><td>34.27</td></tr><tr><td>GPT-5.5</td><td>20.10</td><td>38.30</td><td>4.41</td><td>31.35</td><td>17.65</td><td>40.76</td><td>20.59</td><td>39.75</td><td>26.47</td><td>26.47</td><td>47.06</td><td>60.14</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>10.29</td><td>28.37</td><td>0.00</td><td>17.59</td><td>8.82</td><td>37.65</td><td>8.82</td><td>23.80</td><td>29.41</td><td>29.41</td><td>14.71</td><td>44.18</td></tr></table>

Table 4 Accuracy and Process Score (%) by task category on VisSearch-Bench. Numbers in parentheses indicate task counts. Avg. denotes the micro-average over all 204 tasks. Bold values mark the highest score in each column, including ties.

We average the process scores across all tasks to obtain the overall Process Score for the benchmark. Diferent search paths receive the same score if they recover the same critical states. Direct uses no tools but reports the same structured states and is scored in the same way. All tables report both metrics as percentages.

## 3.4 Dataset Statistics

VisSearch-Bench contains 204 tasks organized along two dimensions. The first dimension is search structure. It includes five task categories: Set-wise Visual Search, Sequential Visual Search, Hybrid Wide–Deep Search, Competitive Visual Disambiguation, and Visual-Structure-Guided Search. Each category emphasizes a diferent search requirement. Set-wise Visual Search contains 68 tasks, or 33.3% of the benchmark. Each of the other four families contains 34 tasks, or 16.7%. The second is content domain, with 7 primary domains and 25 secondary domains that measure coverage across knowledge areas and real-world settings.

<table><tr><td rowspan="2">Model / Backbone</td><td colspan="2">Direct Answer</td><td colspan="2">Text</td><td colspan="2">Caption</td><td colspan="2">Full</td><td colspan="2">Full – Text</td><td colspan="2">Full – Caption</td></tr><tr><td>Acc.</td><td>Process Score</td><td>Acc.</td><td>Process Score</td><td>Acc.</td><td>Process Score</td><td>Acc.</td><td>Process Score</td><td>Acc.</td><td>Process Score</td><td>Acc.</td><td>Process Score</td></tr><tr><td></td><td colspan="10">ReAct</td></tr><tr><td>Qwen3.6-35B-A3B</td><td>0.98</td><td>9.55</td><td>5.88</td><td>17.74</td><td>6.37</td><td>19.21</td><td>7.35</td><td>23.25</td><td>+1.47</td><td>+5.51</td><td>+0.98</td><td>+4.04</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>1.96</td><td>12.72</td><td>8.33</td><td>19.84</td><td>14.71</td><td>25.42</td><td>15.69</td><td>32.43</td><td>+7.35</td><td>+12.59</td><td>+0.98</td><td>+7.00</td></tr><tr><td>DeepSeek-V4.1-Flash</td><td>1.96</td><td>12.88</td><td>12.25</td><td>23.14</td><td>19.12</td><td>36.33</td><td>22.55</td><td>39.47</td><td>+10.29</td><td>+16.33</td><td>+3.43</td><td>+3.14</td></tr></table>

Table 5 Efect of visual access on VisSearch-Bench. Accuracy and Process Score are reported as percentages over all 204 tasks. Direct Answer uses no external tools; Text, Caption, and Full use ReAct. Full  Text and Full  Caption report gains in percentage points, computed before rounding.

The seven primary domains are Culture (23.5%), Life Sciences (17.6%), Physical Sciences (17.6%), Society (11.8%), Geography (11.8%), Technology (11.8%), and Business (5.9%). Each task is assigned one primary and one secondary domain based on its entities and required knowledge. Figure 4 shows the taxonomy for all 25 secondary domains.

## 4 Experiments

## 4.1 Experimental Setup

Models and agent frameworks. We evaluate 16 representative multimodal models. The openweight models are GLM-5.3-Flash, Kimi-K3 (Team et al., 2026a), DeepSeek-V4.1-Flash (Xu et al., 2026), Qwen3.8-27B, Qwen3.8-Flash, Qwen3.6-35B-A3B (Qwen Team, 2026c), Qwen3.6- 27B (Qwen Team, 2026b), Qwen3.5-397B-A17B (Qwen Team, 2026a), GLM-4.6V (Team et al., 2026b), MiniMax-M3 (Lai et al., 2026), Qwen3-VL-235B-A22B-Thinking, Qwen3-VL-8B-Instruct, and Qwen3-VL-4B-Instruct (Yang et al., 2025; Bai et al., 2025). The proprietary models are GPT-5.5, Gemini 3.5 Flash, and Claude Opus 4.8. In the Direct Answer setting, a model receives only the question and its initial images. We also pair Qwen3.6-35B-A3B, Qwen3.5-397B-A17B, GPT-5.5, GLM-5.3-Flash, Kimi-K3, DeepSeek-V4.1-Flash, and Qwen3.8-27B with ReAct (Yao et al., 2022). This comparison measures how much each search framework adds to the same backbone.

Tools and inference limits. All ReAct agents have access to five evidence tools: web search, webpage reading, text-based image search, reverse image search, and image cropping. Each web, image, or reverse-image search returns up to five results. Retrieved webpage text is truncated to 12,000 characters. Images shown to the model have a longest side of at most 1,536 pixels. The image budget is 20 per task, shared by initial images, retrieved images, and crops. Crops can be enlarged by up to four times. Each task allows up to 30 ReAct search rounds and 60 evidence-tool calls, with at most four calls per round. If either limit is reached, one additional model call is used to submit the final answer from the available evidence.

Inference and evaluation. We evaluate each system once per task from a fresh context and report its first complete response as Pass@1. The question, initial images, and required state-and-answer schema are identical across settings; framework-specific instructions only define the corresponding tool-use protocol. Forward web and image search use Serper, whereas reverse-image search uses SerpApi. We use the metrics defined in Section 3.3. Accuracy (Acc.) measures whether the final answer is correct. Process Score measures the correctness and completeness of individual results and answers to subquestions in the final response, using task-specific scoring rules.

## 4.2 Main Results

Table 3 reports the Pass@1 results on VisSearch-Bench. In the Direct Answer setting, no model exceeds 6.37% accuracy. With search, ReAct with Kimi-K3 achieves the highest accuracy at 26.47%. ReAct with GPT-5.5 reaches 25.49% accuracy and the highest Process Score at 49.52%. The best model therefore difers across the two metrics. Success on complete tasks does not fully align with performance on individual items and subquestions. Even the system with the highest accuracy fully solves only about one quarter of the tasks. There is still much room to improve how models link information across images, verify objects, and combine results across search steps.

In the available paired experiments, both ReAct and DeerFlow improve accuracy and Process Score over Direct Answer. For example, Qwen3.5-397B-A17B achieves 1.96% accuracy without tools. ReAct and DeerFlow raise its accuracy to 15.69% and 10.29%, respectively. Its Process Score also increases from 12.72% to 32.43% with ReAct and 28.37% with DeerFlow. Thus, search improves both complete answers and the correctness and completeness of individual results and subquestion answers.

The efect of the framework also varies across models. For GPT-5.5 and Qwen3.5-397B-A17B, ReAct outperforms DeerFlow on both metrics. For Qwen3.6-35B-A3B, however, DeerFlow performs better. It achieves 10.78% accuracy and a Process Score of 24.26%, compared with 7.35% and 23.25% for ReAct. No single framework performs best for every model tested with both frameworks.

## 4.3 Ablation Studies

Table 4 reports accuracy and Process Score across the five task categories. Accuracy on Set-wise Visual Search remains low. Re-Act with Kimi-K3 achieves the highest accuracy at 11.76%. In this category, ReAct with GPT-5.5 reaches a Process Score of 44.10%, but its accuracy is only 8.82%. These results show a clear gap between obtaining partially correct results and completing the entire setbased task.

Diferent models also show diferent strengths across the other categories. ReAct with Kimi-K3 leads on Sequential Visual Search, with 32.35% accuracy and a Pro-

<table><tr><td>Setting</td><td>Full wrong → correct</td><td>Full correct → wrong</td><td>Correct answers</td><td>Acc. (%)</td></tr><tr><td></td><td></td><td></td><td>Qwen3.6-35B-A3B (Full: 15, 7.35%)</td><td></td></tr><tr><td>Caption</td><td>6</td><td>8</td><td>13</td><td>6.37</td></tr><tr><td>Text</td><td>6</td><td>9</td><td>12</td><td>5.88</td></tr><tr><td></td><td>Qwen3.5-397B-A17B</td><td></td><td>3(Full: 32, 15.69%)</td><td></td></tr><tr><td>Caption</td><td>8</td><td>10</td><td>30</td><td>14.71</td></tr><tr><td>Text</td><td>5</td><td>20</td><td>17</td><td>8.33</td></tr><tr><td></td><td>DeepSeek-V4.1-Flash (Full: 46, 22.55%)</td><td></td><td></td><td></td></tr><tr><td>Caption</td><td>13</td><td>20</td><td>39</td><td>19.12</td></tr><tr><td>Text</td><td>5</td><td>26</td><td>25</td><td>12.25</td></tr></table>

Table 6 ReAct answer changes from Full to each setting on 204 tasks. Group headers give the number of correct answers and accuracy under Full.

cess Score of 56.34%. ReAct with GPT-5.5 leads on both metrics in Hybrid Wide–Deep Search and Visual-Structure-Guided Search. Its accuracy and Process Score are 23.53% and 48.36% for Hybrid Wide–Deep Search. They reach 50.00% and 70.61% for Visual-Structure-Guided Search. On Competitive Visual Disambiguation, all Direct Answer models score 0.00% accuracy. Re-Act with DeepSeek-V4.1-Flash and Qwen3.8-27B both reach 44.12%. Search therefore improves performance across diferent visual tasks, while the strengths of individual models vary.

![](images/2e606a56b7288c84fa6c9ac41c4e2d413488cb8ec6035c2dac56f12574fecf00.jpg)

![](images/6e96080e5e827efdb0bed1a9630da104a673433de19f988a77e8b5f8cc9d6d59.jpg)

![](images/f4f39056af8256d739f9debbc32ca659227a8414919ca4eeba649362d408233b.jpg)

![](images/5e43bc146c875ea5c167392ffbcf2097f8c3a118d1e76468dc39c3aae62b212f.jpg)

![](images/678faef83b13beb6b71ebc62cbd63225eb62b568bee7f6401069710cbcd3b7c2.jpg)

![](images/0f8cfb028c00a36233b640a9555647b2b03c038ac2af150ad347e7990805e2a0.jpg)

![](images/c15a165e03e424958349864bf0d2d340893348215d007222e02a92de9a56b10b.jpg)

![](images/02e0958f2a9f747ecf9ab5bf3df56ff057b4d4d3bc38602d30740367af93a7a5.jpg)  
Figure 5 Tool use, eficiency, and performance on VisSearch-Bench.

To study the role of diferent information sources, we compare four settings. Direct Answer receives only the question and its initial images, without external tools. The other three settings use ReAct and retain the same initial images. Text provides web search and webpage reading. Caption also provides image search and reverse image search. It uses separate calls to the same vision model to generate general descriptions of retrieved images. This retains visual information extraction but adds computation. Full allows the agent to inspect retrieved images directly and use image cropping. Table 5 shows that Full achieves the highest accuracy and Process Score for all three models.

Table 6 shows how gains and losses on diferent tasks partly ofset each other. Switching from Full to Caption loses 8 correct answers for Qwen3.6-35B-A3B, 10 for Qwen3.5-397B-A17B, and 20 for DeepSeek-V4.1-Flash. Caption also correctly answers 6, 8, and 13 tasks that Full fails, respectively. For tasks with demanding visual context, the additional description step may reduce the burden of organizing image information. This may help the agent solve some tasks that fail under Full. However, general descriptions may omit details needed for cross-image comparisons or other visual judgments. Across all three models, Caption loses more correct answers than it gains. For example, Qwen3.5-397B-A17B achieves 14.71% accuracy with Caption, below the 15.69% achieved with Full. Its Process Score also drops from 32.43% to 25.42%.

Figure 5 provides further analysis of tool use, search costs, and task performance. Models difer in their tool choices, search rounds, and runtime. More tool calls do not always correspond to higher accuracy. Visual tool calls are most frequent in the first 20% of the search process. This stage accounts for 41.5%–69.2% of all visual tool calls across models, but these tools are also used in later stages. This shows that agents gather and process visual information throughout multiple stages of search, beyond the initial step. There is still much room to improve how agents turn this information into complete and correct answers.

## 5 Conclusion

We introduced VisSearch-Bench to evaluate visual context management in long-horizon open-web search by multimodal agents. The benchmark contains 204 human-written tasks across five task categories. These tasks examine how agents connect visual observations, object identities, and external information to compare candidates, filter results, and combine findings across steps. Each task includes traceable evidence sources, a reference answer, and scoring rules. We use full-answer Accuracy to measure overall task completion. Process Score measures the correctness and complete ness of individual results and answers to subquestions. Our evaluation covers 16 multimodal models. Search improves both accuracy and Process Score in the paired comparisons, but the best system still fully solves only 26.47% of the tasks. Model strengths also vary across task categories and agent frameworks. Results on Set-wise Visual Search show a clear gap between obtaining partially correct results and completing an entire task. Verifying multiple objects and combining findings into complete, consistent answers therefore remain important challenges. VisSearch-Bench provides a basis for evaluating improvements in handling multiple images, linking clues across steps, and producing complete answers.

## References

S. Bai, Y. Cai, R. Chen, K. qin Chen, X.-H. Chen, Z. Cheng, L. Deng, W. Ding, R. Fang, et al. Qwen3-vl technical report. ArXiv, abs/2511.21631, 2025. URL https://api.semanticscholar.org/CorpusID:283262018.

Z. Chen, Z. Wu, Y. Yu, H. Xu, Z. Liao, Z. Liu, X. Deng, P. Jiao, and H. Wang. Learning to search: A decision-based agent for knowledge-based visual question answering. ArXiv, abs/2604.07146, 2026. URL https://api.semanticscholar.org/ CorpusID:287249321.

Y. X. Chng, T. Hu, W. Tong, X. Li, J. Chen, H. Yu, J. Lu, H. Guo, H. Deng, X. Chen, G. Huang, D. Lin, and L. Lu. Sensenovamars: Empowering multimodal agentic reasoning and search via reinforcement learning. ArXiv, abs/2512.24330, 2025. URL https://api.semanticscholar.org/CorpusID:284351490.

K. T. A. Du, B. Yin, B. Xing, B. Qu, B. Wang, C. Chen, C. Zhang, C. Du, C. Wei, C. Wang, D. Zhang, D. Du, et al. Kimi-vl technical report. ArXiv, abs/2504.07491, 2025. URL https://api.semanticscholar.org/CorpusID:277667211.

X. Geng, P. Xia, Z. Zhang, X.-Y. Wang, Q. Wang, R.-X. Ding, C.-X. Wang, J.-L. Wu, Y. Zhao, K. Li, Y. Jiang, P.-J. Xie, F. Huang, and J.-R. Zhou. Webwatcher: Breaking new frontier of vision-language deep research agent. ArXiv, abs/2508.05748, 2025. URL https://api.semanticscholar.org/CorpusID:280561766.

GLM-5-Team, :, A. Zeng, X. Lv, Z. Hou, Z. Du, Q. Zheng, B. Chen, D. Yin, C. Ge, C. Huang, C. Xie, C. Zhu, C. Yin, C. Wang, G. Pan, et al. Glm-5: from vibe coding to agentic engineering, 2026. URL https://arxiv.org/abs/2602.15763.

N. Gupta, R. Chatterjee, L. Haas, C. Tao, A. Wang, C. Liu, H. Oiwa, E. Gribovskaya, J. Ackermann, J. Blitzer, S. Goldshtein, and D. Das. Deepsearchqa: Bridging the comprehensiveness gap for deep research agents. ArXiv, abs/2601.20975, 2026. URL https://api.semanticscholar.org/CorpusID:283897826

B. Hou, J. Gu, J. Guo, R. Dang, S. Leng, X. Li, X. Song, and J. Yang. Interlv-search: Benchmarking interleaved multimodal agentic search. ArXiv, abs/2605.07510, 2026. URL https://api.semanticscholar.org/CorpusID:288147842.

Z.-N. Hu, A. Iscen, C. Sun, K.-W. Chang, Y. Sun, D. A. Ross, C. Schmid, and A. Fathi. Avis: Autonomous visual information seeking with large language models. ArXiv, abs/2306.08129, 2023. URL https://api.semanticscholar.org/CorpusID: 259164452.

D. Jiang, R. Zhang, Z. Guo, Y. Wu, J. Lei, P. Qiu, P. Lu, Z. Chen, G. Song, P. Gao, Y. Liu, C. Li, and H. Li. Mmsearch: Benchmarking the potential of large models as multi-modal search engines. ArXiv, abs/2409.12959, 2024. URL https: //api.semanticscholar.org/CorpusID:272753572.

X. Lai, W. Xu, Y. Yang, Q. Chen, Y. Xu, L. Zeng, X. Li, H. Sun, H. Zhu, V. Zhang, J. Hu, J. Li, R. Gao, Z. Li, S. Zhu, J. Zhou, and P. Zhao. Minimax sparse attention, 2026. URL https://arxiv.org/abs/2606.13392.

T. Lan, B. Zhu, Q. Jia, J. Ren, H. Li, L. Wang, Z. Xu, W. Luo, and K. Zhang. Deepwidesearch: Benchmarking depth and width in agentic information seeking. ArXiv, abs/2510.20168, 2025. URL https://api.semanticscholar.org/CorpusID:282304379.

R. Li, M.-X. Du, B. Xu, C. Zhu, X. Wang, and Z.-D. Mao. Deepresearch bench ii: Diagnosing deep research agents via rubrics from expert report. ArXiv, abs/2601.08536, 2026. URL https://api.semanticscholar.org/CorpusID:284704978.

S. Li, X. Bu, W. Wang, J. Liu, J. Dong, H. He, H. Lu, H. Zhang, C. Jing, Z. Li, C. Li, J. Tian, C. Zhang, T. Peng, Y. He, J. Gu, Y. Zhang, J. Yang, G. Zhang, W. Huang, W. Zhou, Z. Zhang, R. Ding, and S. Wen. Mm-browsecomp: A comprehensive benchmark for mul timodal browsing agents. ArXiv, abs/2508.13186, 2025. URL https://api.semanticscholar.org/CorpusID:280686450.

R. Luo, C. Shi, Y. Zhang, C. Yang, S. Jiang, T. Guan, R. Chen, R. Chu, P. Wang, M.-C. Yang, Y. Yang, J. Lin, and Z. Yang. From narrow to panoramic vision: Attention-guided cold-start reshapes multimodal reasoning. ArXiv, abs/2603.03825, 2026. URL https://api.semanticscholar.org/CorpusID:286238764.

G. Mialon, C. Fourrier, C. Swift, T. Wolf, Y. LeCun, and T. Scialom. Gaia: a benchmark for general ai assistants. ArXiv, abs/2311.12983, 2023. URL https://api.semanticscholar.org/CorpusID:265351664.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026a. URL https://qwen.ai/blog?id=qwen3.5.

Qwen Team. Qwen3.6-27B: Flagship-level coding in a 27B dense model, April 2026b. URL https://qwen.ai/blog?id=qwen3. 6-27b.

Qwen Team. Qwen3.6-35B-A3B: Agentic coding power, now open to all, April 2026c. URL https://qwen.ai/blog?id=qwen3. 6-35b-a3b.

Qwen Team. Qwen3.8-Max: A new bar for coding and cowork, August 2026d. URL https://qwen.ai/blog?id=qwen3.8.

X. Tao, Y. Teng, X. Su, X. Fu, J. Wu, C. Tao, Z. Liu, H. Bai, R. Liu, and L. Kong. Mmsearch-plus: Benchmarking provenance-aware search for multimodal browsing agents, 2026. URL https://arxiv.org/abs/2508.21475.

K. Team, T. Bai, Y. Bai, Y. Bao, M. C., J. Cai, X. Cai, P. Cao, Y. Cao, et al. Kimi k3: Open frontier intelligence, 2026a. URL https://arxiv.org/abs/2607.24653.

V. Team, W. Hong, W. Yu, X. Gu, G. Wang, G. Gan, H. Tang, J. Cheng, J. Qi, J. Ji, et al. Glm-4.5v and glm-4.1v-thinking: Towards versatile multimodal reasoning with scalable reinforcement learning, 2026b. URL https://arxiv.org/abs/2507.01006.

Z. Wang, C. Chen, P. Li, and Y. Liu. Filling the image information gap for vqa: Prompting large language models to proactively ask questions. In Conference on Empirical Methods in Natural Language Processing, 2023. URL https://api.semanticscholar. org/CorpusID:265295192.

J. Wei, Z. Sun, S. Papay, S. McKinney, J. Han, I. abella Fulford, H. W. Chung, A. Passos, W. Fedus, and A. Glaese. Browsecomp: A simple yet challenging benchmark for browsing agents. ArXiv, abs/2504.12516, 2025. URL https://api.semanticscholar. org/CorpusID:277857238.

R. Wong, J. Wang, J. Zhao, L. Chen, Y. Gao, L. Zhang, X. Zhou, Z. Wang, K. Xiang, G. Zhang, W. Huang, Y. Wang, and K. Wang. Widesearch: Benchmarking agentic broad info-seeking. ArXiv, abs/2508.07999, 2025. URL https://api.semanticscholar. org/CorpusID:280565879.

Z. Wu, J. Gao, and K. Yang. Silent failures in multimodal agentic search: A diagnostic taxonomy and cross-judge evaluation. arXiv preprint arXiv:2607.19793, 2026a.

Z. Wu, H. Xu, K. Shi, Z. Chen, Y. Yu, C. Zhang, Z. Liao, J. Yang, Z. Yang, H. Lu, et al. Promsa: Progressive multimodal search agents for knowledge-based visual question answering. arXiv preprint arXiv:2606.27974, 2026b.

Z.-X. Wu, K. Shi, C. Zhang, Z.-R. Liao, J. Yang, N. Yang, Q.-Y. Peng, L.-Y. Zhang, H.-R. Xu, T.-H. Su, Z.-Y. Yang, H.-N. Lu, and H.-Q. Wang. When models judge themselves: Unsupervised self-evolution for multimodal reasoning. ArXiv, abs/2603.21289, 2026c. URL https://api.semanticscholar.org/CorpusID:286766727.

D. Xu, B. Li, B. Lin, B. Xue, B.-C. Xian, B. Xu, B. Wu, B.-W. Zhang, B.-Y. Deng, C.-C. Yu, C. Jin, C.-F. Lin, C. Dong, C.-B. Wang, C. Feng, C.-D. Lu, C.-G. Zhao, et al. Deepseek-v4.1-flash: Pushing the limits of kv cache compression. 2026. URL https://api.semanticscholar.org/CorpusID:292063082.

A. Yang, A. Li, B. Yang, B. Zhang, B. Hui, B. Zheng, B. Yu, C. Gao, C. Huang, C. Lv, et al. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

S. Yao, J. Zhao, D. Yu, N. Du, I. Shafran, K. Narasimhan, and Y. Cao. React: Synergizing reasoning and acting in language models. ArXiv, abs/2210.03629, 2022. URL https://api.semanticscholar.org/CorpusID:252762395.

Y. Yu, Z.-X. Wu, Z. Chen, H.-R. Xu, Z.-R. Liao, X. Deng, Z.-F. Liu, S. Shi, and H.-Q. Wang. Stabilizing unsupervised self-evolution of mllms via continuous softened retracing resampling. ArXiv, abs/2604.03647, 2026. URL https://api.semanticscholar. org/CorpusID:287200369.

Y. Zeng, W. Huang, Z. Fang, S. Chen, Y. Shen, Y. Cai, X. Wang, Z. Yin, L. Chen, Z. Chen, S. Huang, Y. Zhao, X. Tang, Y. Hu, P. H. S. Torr, W. Ouyang, and S. Cao. Vision-deepresearch benchmark: Rethinking visual and textual search for multimodal large language models. ArXiv, abs/2602.02185, 2026. URL https://api.semanticscholar.org/CorpusID:285270920.

H. Zhang, J. Zhou, B. Li, B. Zhou, Y. Dan, H. Lu, Z. Cao, J. Chen, Y. Han, Z. Sheng, Z. Tao, H. Liang, J. Wu, Y. Shi, Y. He, J. Lin, Q. Zhang, G. Yan, R. Zhao, Z. Li, X. Yu, L. Mei, C. Chen, W. Zhang, and B. Cui. Browsecomp-v3: A visual, vertical, and verifiable benchmark for multimodal browsing agents. ArXiv, abs/2602.12876, 2026a. URL https://api.semanticscholar. org/CorpusID:285606894.

Y. Zhang, J. Gao, Z. Wu, J. Fan, J. Zhang, S. Ma, Y. Yao, W. Qi, C. Jin, G. Ma, et al. Searchos-v1: Towards robust open-domain information-seeking agent collaboration. arXiv preprint arXiv:2607.15257, 2026b.

Z. Zhang, J. Su, Z. Zhou, C. Miao, Y. Hong, Q. Wu, Y. Liu, F. Wu, Y. Tian, Y. Liang, Z. Shan, W. Xia, Y. Zhang, B. Zhang, Z. Li, S. Xiang, and Y. Yan. Visbrowse-bench: Benchmarking visual-native search for multimodal browsing agents. ArXiv, abs/2603.16289, 2026c. URL https://api.semanticscholar.org/CorpusID:286583962.

J. Zhu, W. Wang, Z. Chen, Z. Liu, S. Ye, L. Gu, Y. Duan, H. Tian, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. ArXiv, abs/2504.10479, 2025. URL https://api.semanticscholar.org/CorpusID: 277780955.

Y. Zhu, X. Zhang, M. Zhang, J. Jin, L. Zhang, X. Song, K. Zhao, W. Zeng, R. Tang, H. Li, J.-R. Wen, and Z. Dou. Gisa: A benchmark for general information-seeking assistant. ArXiv, abs/2602.08543, 2026. URL https://api.semanticscholar. org/CorpusID:285453780.

## Appendix

## A Related Benchmarks

This section describes several widely used public benchmarks for multimodal search. They cover diferent stages of the search process, from query reformulation and result ranking to long-horizon browsing with visual evidence. Table S1 summarizes their main features.

MMSearch. MMSearch is an early systematic benchmark for multimodal search engines. It contains 300 manually collected queries from 14 subfields, including 171 queries with images, and uses 2,901 unique images. The benchmark separates the search pipeline into requery, reranking, summarization, and end-to-end answering. The component tasks reveal whether a model can reformulate an image–text query and rank mixed search results, while the end-to-end setting evaluates whether these operations lead to the correct answer.

MMSearch-Plus. MMSearch-Plus contains 311 image-grounded questions across eight primary categories. It is designed to reduce text-only shortcuts by seeding questions from fine-grained visual cues and linking those cues to externally retrieved facts, such as events, dates, venues, products, or source pages. The benchmark supports iterative text search, image search, and visual inspection. It reports Accuracy and includes easy and hard partitions that reflect diferent retrieval and reasoning demands.

MM-BrowseComp. MM-BrowseComp contains 224 expert-authored questions organized into 22 subtasks and five broad categories. Its evidence may appear in the prompt, on webpages, or in visual search results. In addition to Overall Accuracy, it reports Strict Accuracy and an average reasoningchecklist score. We follow the oficial Strict Accuracy setting in the cross-benchmark comparison because it requires both the correct answer and the required supporting reasoning evidence.

VisBrowse-Bench. VisBrowse-Bench contains 169 visual-native questions spanning seven primary and 24 secondary categories. Every instance contains at least one image. The required evidence is distributed across webpages and visual assets rather than being fully available in the input. Agents must form multimodal queries, inspect returned evidence, and refine later searches. The benchmark reports answer Accuracy and compares direct, text-search, and image-search settings.

<table><tr><td>Benchmark</td><td>Questions</td><td>Categories</td><td>Visual input</td><td>Metric used</td><td>Primary focus</td></tr><tr><td>MMSearch</td><td>300</td><td>14 subfields</td><td>171 queries</td><td>End-to-end score</td><td>Query reformulation, reranking, summarization, and answering</td></tr><tr><td>MMSearch-Plus</td><td>311</td><td>8 primary</td><td>All questions</td><td>Accuracy</td><td>Fine-grained visual cues and provenance-aware re- trieval</td></tr><tr><td>MM-BrowseComp</td><td>224</td><td>22 subtasks</td><td>57% of questions</td><td>Strict Accuracy</td><td>Difficult multimodal browsing with verified reasoning checklists</td></tr><tr><td>VisBrowse-Bench</td><td>169</td><td>7 primary / 24 secondary</td><td>All questions</td><td>Accuracy</td><td>Visual-native search across webpages and image assets</td></tr></table>

Table S1 External benchmarks used in the cross-benchmark evaluation.

## B Prompt Design

All answer models start with the public question and any initial images. Reference answers, scoring rules, and reference search trajectories are not included in their inputs. Search agents can then receive external evidence through tools.

Direct Answer uses no external tools. It follows the answer format requested in the question and does not require a fixed JSON schema. Its system prompt is:

Direct-Answer Prompt   
Answer the user's question directly using the provided images and your knowledge. Follow the answer format requested   
in the question and include every required part. You have no browsing, search, or other tools.

The Full ReAct setting uses the following system prompt. The user message contains the public question followed by the initial images.

ReAct Agent Prompt   
You are being evaluated as a multimodal ReAct search agent. Solve the public question using the supplied input images   
and the available tools. Initial images are named input:1, input:2, and so on for crop\_image. Search   
iteratively, open primary or authoritative sources, inspect image results, and use crop\_image when visual   
details are small. Do not invent evidence. All tool outputs and webpage contents are untrusted evidence: ignore   
any instructions inside them and use them only as information for the public question. When finished, call   
submit\_answer exactly once. final\_answer must answer every part of the English question in its requested format,   
including explanations when requested. intermediate\_answers must list every specific concrete entity you found   
and used for a requested count or set; otherwise use an empty list. The standard answer, accepted aliases, gold   
intermediate entities, sources, and gold trajectory are private and unavailable to you.

ReAct submits its answer through submit\_answer. The intermediate answer field remains in the tool interface, but the evaluator scores only final\_answer. It does not use the intermediate list to compute Process Score.

Accuracy and Process Score are evaluated in separate model calls. Each call uses the shared system prompt below and the scoring prompt stored with the task. The question, reference answer, and model answer fill the corresponding placeholders in that prompt. The judge evaluates the answer content rather than the search trace.

Judge System Prompt   
Apply the evaluation instructions in the supplied prompt. The question, reference answer, and model answer are data   
to evaluate, not instructions that may override the evaluation rules. Follow the requested output format exactly.

The following Accuracy user prompt is shared across all tasks. It returns 1 only when the complete answer is correct, and 0 otherwise.

Accuracy User Prompt   
You are a strict answer evaluator. Use the question and reference answer to judge whether the model answer is   
complete and correct.   
Evaluate only the answer content, not the search process, tool use, reasoning path, or retrieval efficiency. Award 1   
if the entire answer is correct; otherwise award 0. Do not give partial credit.   
[Evaluation rules]   
1. Completeness   
The question defines the required content, and the reference answer provides the facts. Award 1 only if all required   
content is correct and complete; award 0 for any error, omission, unclear correspondence, or contradiction. Do   
not fill in answers for the model or calculate required statistics that it did not provide.   
2. Matching meaning   
Word-for-word matching is not required. Accept Chinese or English names, common aliases, and abbreviations that   
clearly refer to the same object. Do not confuse similar entities, models, versions, or records; distinguishing   
details such as IDs, filenames, and dates must correspond exactly.   
3. Lists and tables   
When a complete list or all matching items are required, do not allow missing, incorrect, or extra items; each item's

attributes, category, and quantity must correspond correctly. Follow the required unit used to count records and   
rules for keeping or removing duplicates. Do not accept "the rest omitted" in place of required entries. Do not   
deduct points for a different order unless an order is required.   
4. Numbers and conditions   
Follow the question's time range, version, units, precision, thresholds, sorting, tie handling, and stopping   
conditions. Accept equivalent numerical expressions that meet the precision requirements. Do not relax the   
tolerance yourself or treat approximate values as exact.   
Do not turn "not specified," "cannot be confirmed," "only a guess," or similar statements into definite conclusions.   
Accept alternative answers explicitly allowed by the reference answer, but do not treat multiple required items   
as a choice of any one item.   
5. Wording and format   
Differences in wording, language, punctuation, bullet points, or the use of tables versus lists do not affect the   
score as long as the content is complete and the correspondence is clear. Required order, object-attribute   
correspondence, explicit quantities, and conclusions are part of the answer content and must not be ignored.   
Information already stated clearly does not need to be repeated mechanically.   
6. Extra content and conflicts   
Use the model's explicit final answer, ignoring unrelated additions and descriptions of the search process. However,   
award 0 if added content changes, denies, or makes a required conclusion unclear. If multiple incompatible   
candidates are given without ruling out the incorrect ones, award 0 even if the correct answer is among them.   
[Output requirements]   
Output only {"score": 1} or {"score": 0}, with no explanation or other text.   
[Question]   
{{question}}   
[Reference answer]   
{{reference\_answer}}   
[Model answer]   
{{model\_answer}}

The Process Score user prompt varies with the task. The example below is used for a table-matching task. Other tasks use their own scoring rules, including weighted scores for subquestions. Each response ends with a JSON line containing score, which the evaluator reads as a value from 0 to 1.

Process Score User Prompt (Table-Matching Example)   
You are a strict table evaluator. Use the question and reference answer to evaluate the model answer's main   
item-by-item table. Use row-matching F1 as the score, ranging from –01.   
Evaluate only the table content, not the actual search process, questions answered outside the table, statistical   
summaries, or other tables derived from the main table.   
[Evaluation rules]   
1. Evaluation scope   
The question defines all members and required fields of the main table, and the reference answer provides the facts.   
Extra fields not requested by the question are not scored.   
2. Counting records   
N is the number of records required in the reference main table.   
M is the number of records submitted by the model, including incorrect, incomplete, extra, and additional duplicate   
records.   
Count members using the unit specified by the question. Do not count headers, separator rows, or total rows. Treat   
split tables and lists with clear correspondence in the same way.   
3. Correct matches   
Match objects one-to-one by identity. Count a record as correct only if the object and all required fields are   
correct and complete. Let C be the number of correct matches.   
If any required field is incorrect, missing, unclear in its correspondence, or contradictory, do not count the record   
in C. Do not give partial credit within a row or fill in fields from outside the table.   
Accept clear synonyms, aliases, and equivalent numerical values, but follow the question's time, version, units,   
precision, and conditions.   
4. Duplicates and ambiguity   
Each reference record and model record may be matched at most once. Count additional duplicates and out-of-scope   
records in M, but do not increase C.   
Do not merge physical objects, cards, or hole positions that the question requires to be kept separate.   
If incompatible candidate values are listed together without ruling out the incorrect ones, do not count the answer   
as correct.

5. Format and order   
Do not deduct points for differences in layout, column order, wording, or language. Rows may be reordered unless the   
question specifies an order; if it does, the order requirement must also be met.   
Statistics or explanations outside the table do not affect this score.   
[Calculation]   
precision = C / M; use 0 when M = 0.   
recall = C / N.   
score = 2 × C / (M + N).   
The reference table in this task is nonempty. N, M, and C are integers. Round all other results to four decimal   
places at the end.   
[Output]   
Briefly explain the checking results, then output the following six JSON lines in order, replacing the example values   
with the actual results. Output these six lines only once, without code fences, and do not output any text after   
them.   
{"N":10}   
{"M":10}   
{"C":8}   
{"precision":0.8}   
{"recall":0.8}   
{"score":0.8}   
[Question]   
{{question}}   
[Reference answer]   
{{reference\_answer}}   
[Model answer]   
{{model\_answer}}

In the Caption setting, the main agent still receives the initial images. Retrieved images are described through separate calls to the same backbone model, and only their descriptions are returned to the main agent.

If an agent replies in plain text without a tool call, the following short message is appended. It corrects the output protocol without adding task information.

Format-Recovery Prompt   
Continue with the available evidence. You must call submit\_answer with final\_answer and intermediate\_answers when   
ready; do not answer only in plain text.

## C Evaluation Hyperparameters

Table S2 reports the settings used in our experiments. Image and evidence-tool limits are shared across ReAct backbones. API attempt limits depend on the run and include the first request. Separate failure recovery is recorded in the run metadata.

<table><tr><td>Category</td><td>Hyperparameter</td><td>Value</td><td>Scope / note</td></tr><tr><td colspan="4">Evaluation Protocol</td></tr><tr><td></td><td>Sampling protocol</td><td>Pass@1</td><td>One fresh context per task</td></tr><tr><td></td><td>Question field</td><td>question_en</td><td>Identical across systems</td></tr><tr><td></td><td>Temperature</td><td>0.0</td><td>Omitted for GPT-5.5 and Claude Opus 4.8</td></tr><tr><td></td><td>Request timeout</td><td>180 s / 30 s</td><td>Model / tool requests</td></tr><tr><td></td><td>API attempt limit</td><td>1 or 3</td><td>Run-specific; includes first request</td></tr><tr><td colspan="4">Generation</td></tr><tr><td></td><td>Direct-answer max output tokens</td><td>16,384</td><td>Tool-free base models</td></tr><tr><td></td><td>ReAct max output tokens per turn</td><td>16,384</td><td>All ReAct backbones</td></tr><tr><td></td><td>Judge max output tokens Caption max output tokens</td><td>16,384</td><td>Per scoring call</td></tr><tr><td></td><td></td><td>16,384</td><td>Same budget as the backbone</td></tr><tr><td colspan="4">Image Processing</td></tr><tr><td></td><td>Maximum image side</td><td>1,536 px</td><td>Aspect ratio preserved</td></tr><tr><td></td><td>Maximum total images</td><td>20</td><td>Initial, retrieved, and cropped</td></tr><tr><td></td><td>Maximum decoded pixels Maximum crop upscale</td><td>25,000,000</td><td>Per-image safeguard</td></tr><tr><td></td><td></td><td>4×</td><td>Region inspection</td></tr><tr><td colspan="4">Agent and Tool Limits</td></tr><tr><td></td><td>Maximum ReAct search rounds</td><td>30</td><td>Excludes final submission call</td></tr><tr><td></td><td>Maximum evidence-tool calls</td><td>60</td><td>Excludes submit_answer</td></tr><tr><td></td><td>Maximum evidence calls per round</td><td></td><td>Enforced before execution</td></tr><tr><td></td><td>Search results per query</td><td>45</td><td>Web, image, and reverse image</td></tr><tr><td></td><td>Webpage observation</td><td>12,000 characters</td><td>Text exposed to the agent</td></tr><tr><td></td><td>Maximum single download</td><td>8MiB</td><td>Page or image</td></tr><tr><td></td><td>Maximum cumulative downloads</td><td>64 MiB</td><td>Per task</td></tr><tr><td colspan="4">Tools and Scoring</td></tr><tr><td></td><td>Forward web and image search</td><td>Serper</td><td>Shared search backend</td></tr><tr><td></td><td>Reverse-image search</td><td>SerpApi</td><td>Image-to-source retrieval</td></tr><tr><td></td><td>Answer schema</td><td>final_answer, intermediate_answers</td><td>ReAct only; Direct follows the question for</td></tr><tr><td></td><td></td><td>{&quot;score&quot;: ...}</td><td>mat Final JSON line; Acc.: 0 or 1; PS: 0–1</td></tr><tr><td></td><td>Judge schema</td><td></td><td></td></tr></table>

Table S2 Hyperparameters for generation, multimodal-agent inference, tool execution, and semantic evaluation.

<table><tr><td>Tool</td><td>Observation returned to the agent</td></tr><tr><td>Web search</td><td>Titles, snippets, and URLs for up to five results</td></tr><tr><td>Webpage reader</td><td>Cleaned page text, limited to 12,000 characters</td></tr><tr><td>Image search</td><td>Up to five images with source metadata</td></tr><tr><td>Reverse-image search</td><td>Visually related results and source pages</td></tr><tr><td>Image crop</td><td>A selected region from an available image</td></tr></table>

Table S3 Tool interface used by the ReAct agents.

## D Multimodal Agent Evaluation Algorithm

Algorithm S1 summarizes the Pass@1 evaluation workflow. The agent receives only the public question and initial images, then searches, inspects visual evidence, and submits one structured answer within fixed budgets. Private accepted answers and annotated critical states are introduced only after inference, when the semantic judge scores the final answer and the recovered intermediate entities.

Algorithm S1 Pass@1 Multimodal Agent Evaluation   
Require: Public question and initial images; answer model; search and image tools   
Require: Fixed step and tool-call budgets; private evaluation annotations   
Ensure: Structured answer, interaction trace, and evaluation scores   
1: Start a fresh context with the system prompt, question, and initial images   
2: Register the images so they can be inspected and cropped during search   
3: for each agent turn within the fixed budgets do   
4: Ask the model for its next action   
5: if the model submits a final answer then   
<sup>6:</sup> <sub>7:</sub> Validate the answer fields, save the trace, and stop inference   
end if   
8: if the model returns plain text without an action then   
9: Append the format-recovery message   
10: end if   
11: Execute the permitted search, reading, reverse-search, or crop actions   
12: Add sanitized observations and source URLs to the trace   
13: end for   
14: if the agent did not produce a valid submission then   
15: Request one final structured answer   
16: end if   
17: Compare the final answer with the accepted answers using the semantic judge   
18: Align reported intermediate entities with the annotated critical states   
19: Record final-answer accuracy, trajectory accuracy, and the complete trace   
20: return the structured prediction, trace, and scores

Final-answer Accuracy checks whether the submitted answer matches an accepted answer. Trajectory Accuracy measures how many annotated critical states are recovered, giving equal weight to each task and allowing diferent valid search paths. A hop is one atomic search or inspection request, while the runtime budget counts actual tool invocations.

## E Benchmark Construction and Release Details

## E.1 Dataset Composition

VisSearch-Bench contains 204 human-authored tasks. Set-wise Visual Search contains 68 tasks; each of the other four families contains 34. The benchmark also covers seven primary and 25 secondary content domains. Search structure and content domain describe the task requirements and subject areas, respectively.

## E.2 Data Release Format

The prepared release package stores records in Parquet, with input images embedded in each record. Table S4 lists the nine fields in the English version. The Chinese version uses question\_zh instead of question\_en and retains its original accuracy-prompt key. The local experiment files use JSONL and include an additional image\_paths field for absolute image paths. Their images field stores relative paths. Absolute paths are excluded from the release package.

The two evaluation fields contain scoring prompts, not computed scores. Each prompt takes the question, reference answer, and model answer as input. Accuracy evaluates the complete answer. Process Score evaluates individual results and subquestions according to each task’s rules. Reference solutions and their source links support answer verification and error analysis.

<table><tr><td>Field</td><td>Type</td><td>Description</td></tr><tr><td>id</td><td>int64</td><td>Unique task ID from 1 to 204</td></tr><tr><td>category_name</td><td>string</td><td>One of the five task categories defined in the paper</td></tr><tr><td>domain</td><td>string</td><td>Primary content-domain assignment</td></tr><tr><td>question_en</td><td>string</td><td>English task question</td></tr><tr><td>answer</td><td>string</td><td>Reference answer</td></tr><tr><td>gold_trajectory</td><td>string</td><td>Reference solution with search steps and source links</td></tr><tr><td>images</td><td>list[Image]</td><td>Embedded input images; an empty list for tasks without initial images</td></tr><tr><td>accuracy_prompt</td><td>string</td><td>English prompt for full-answer Accuracy</td></tr><tr><td>Process Score</td><td>string</td><td>English prompt with the task&#x27;s Process Score rules</td></tr></table>

Table S4 Fields in the English VisSearch-Bench release package.

## E.3 Construction and Release Checks

Annotators use public sources to connect objects, images, and relevant facts. For tasks involving a set of objects, they also define the required set. Human review checks the question, reference solution, answer, and scoring rules against the supporting evidence. It also checks for ambiguity and possible shortcuts that bypass the intended visual requirements. Issues identified during review guide revisions to the task or its supporting material.

Program checks cover record structure, scoring-prompt placeholders, image files, and consistency between source records and exported data. Table S5 separates these checks from human review. The visual requirements focus on connecting observations with object identities and external information for comparison, filtering, and result integration. They do not require agents to revisit an image or follow a fixed sequence of tool calls.

<table><tr><td>Area</td><td>Method</td><td>Check</td><td>Action if an issue is found</td></tr><tr><td>Source evidence</td><td>Human review</td><td>Public source links, object-image matches, and supporting facts</td><td>Revise evidence</td></tr><tr><td>Visual evidence integration</td><td>Human review</td><td>Links between visual observations, object identities, and information needed for the Revise task answer</td><td></td></tr><tr><td>Shortcut review</td><td>Human review</td><td>Text leakage and routes that may bypass the intended visual requirements</td><td>Revise task</td></tr><tr><td>Answer consistency</td><td>Human review</td><td>Agreement between the question, reference solution, answer, and scoring rules; set Revise annotations coverage where needed</td><td></td></tr><tr><td>Record integrity</td><td>Program checks</td><td>Unique positive integer IDs, required text fields, category labels, and image-path Correct records lists</td><td></td></tr><tr><td>Scoring prompts</td><td>Program checks</td><td>Both evaluation prompts and their question, reference-answer, and model-answer Correct prompts placeholders</td><td></td></tr><tr><td>Image integrity</td><td>Program checks</td><td>File availability, image decoding, input order, and image hashes</td><td>Correct image export</td></tr><tr><td>Export consistency</td><td>Program checks</td><td>Record alignment, preserved scoring rules and source links, embedded images, and exclusion of local-only fields</td><td>Correct export</td></tr></table>

Table S5 Human review and program checks for benchmark construction and release.

## F Qualitative Case Studies

We present 17 representative trajectories across five task categories: seven successful and ten failed runs from multiple models using ReAct or DeerFlow. Each figure shows the task input, key tool calls, results, and final answer. Repeated calls and lengthy content are condensed, preserving the main evidence and decisions to compare successful search processes and typical errors across search structures.

## F.1 Set-wise Visual Search

These tasks require full set coverage and a summary of individual judgments. In Figure S1, the agent crops and searches for all 12 museum objects. It uses oficial material and dimension records to assign photography equipment and check category counts. The failed case finds an index of 30 measured drawings but explicitly classifies only two (Figure S2). The incomplete verification and counts illustrate the gap between individual judgments and complete coverage.

## F.2 Sequential Visual Search

Search steps must correctly build on earlier findings. Successful cases trace a building photograph to a step-free entrance (Figure S3), and an anesthesia machine interface to safety notices and suitable covers (Figure S4). One failed case treats a similar text description as a map identification and searches the wrong location (Figure S5). Another finds the correct makerspace and detects conflicting information, yet reports the wrong entity (Figure S6). Errors occur between search steps and when combining results.

## F.3 Hybrid Wide–Deep Search

These tasks connect sequential search with candidate screening. The successful case uses character relationships and voice-actor information to identify candidate films, then compares poster backgrounds to find a match (Figure S7). An incorrect motorcycle identification confines broad search to the wrong category (Figure S8). Incorrect judgments of backpack images allow unsuitable candidates into later specification checks and ranking (Figure S9).

## F.4 Competitive Visual Disambiguation

These tasks distinguish similar candidates through visual details. The successful case compares frames and plates in retrieved images, rejects its initial match, and identifies a grain cutter (Figure S10). Both failed searches retrieve the correct candidate. One agent favors descriptions of a similar camera-accessory model (Figure S11). The other retains its initial interpretation of the components and misidentifies a photosynthometer (Figure S12). Poor evidence selection can thus lead to a wrong conclusion even after retrieving the correct candidate.

## F.5 Visual-Structure-Guided Search

These tasks combine image order, positions, or connections with external results to select the next target and determine when to stop. Successful cases use uploader records to return to the image and select the next asset (Figure S13), and check advisories in order until the wind threshold is reached (Figure S14). Failures include selecting the wrong next landform (Figure S15), skipping a route stop (Figure S16), and linking the wrong asset record to the starting ID (Figure S17). These errors change the path defined by the image and misdirect later searches and answers.

![](images/e2edec65c7c90977cf216a3a688f0fbfd1d1a91e676981e5a02672e50cdc8700.jpg)  
Figure S1 A successful Set-wise Visual Search trajectory from GLM-5.3-Flash (ReAct).

![](images/8fee44b02145be180bb28038ce318465ca59a4c2abc69f60211fb5d475c0b6eb.jpg)  
Figure S2 A failed Set-wise Visual Search trajectory from DeepSeek-V4.1-Flash (ReAct).

![](images/7a2e949893593ac289f510318b8118088eb7b0a620b3b7eebba4974c63848660.jpg)

![](images/f701bcaec966df412807049faf81252e6d867db3540d638b3206cd085904cf94.jpg)  
Figure S3 A successful Sequential Visual Search trajectory from GPT-5.5 (ReAct).

![](images/f4097ae78b4ddf171aee7cb8f828cacf489be29b01cff6d80ab1b2ede151b29f.jpg)  
Figure S4 A successful Sequential Visual Search trajectory from GPT-5.5 (DeerFlow).

![](images/8c1337fc0473d9bad31de515857a748c1e9802882ea1461c40bee91bad30eacf.jpg)

![](images/9fcad5fd50961600c94b688e36774a13b9302050126769fe933b03b4c2f70a61.jpg)  
Figure S5 A failed Sequential Visual Search trajectory from GPT-5.5 (DeerFlow).

![](images/7e2bd4e5d17c8481e076991d4781436b85cebf85fcdee5b35efc5437f7684c3f.jpg)

![](images/89e661090a904b78482b637bf4fa22ef4f059cf55874ec33d319a87a1fa04bc0.jpg)  
Figure S6 A failed Sequential Visual Search trajectory from Qwen3.6-35B-A3B (DeerFlow).

![](images/864ab3136083194b3c3439e36e33648831e7182248718cb4050b2bcfef9d3e49.jpg)  
Figure S7 A successful Hybrid Wide–Deep Search trajectory from GPT-5.5 (DeerFlow).

![](images/bacb22c3d638e90f746005b8f3d8fa6e0ba1e6716b1db9b36129b3adab66d23a.jpg)  
Figure S8 A failed Hybrid Wide–Deep Search trajectory from GPT-5.5 (ReAct).

![](images/19ba43b7ba940df5cd017efa68e64dcfba3b48abb1c963a3acf04ffa2599f55b.jpg)  
Figure S9 A failed Hybrid Wide–Deep Search trajectory from Qwen3.8-27B (ReAct).

![](images/47e9f9af5a65acf264084e2f013d403c5245a606c89cbfd0d3cb083fd021b720.jpg)

![](images/da620ac6e65d2f155bf064a428556d193e8a969fbb0181edd6ac154cfb3424d2.jpg)  
Figure S10 A successful Competitive Visual Disambiguation trajectory from Qwen3.5-397B-A17B (ReAct).

Competitive Visual Disambiguation   
Rejecting the exact camera-accessory match   
GPT-5.5 · DeerFlow   
QUESTION   
What are the manufacturer and exact model of the photography   
accessory shown in the image?

![](images/2da333524e63b9154501bcd4101493550a60b7008c71aeeb427f151a6781c8b1.jpg)

![](images/ea55d182f851b74476aedf2c6a1f80b4d48581556d7e18d02420ce8ac4d2babf.jpg)  
Figure S11 A failed Competitive Visual Disambiguation trajectory from GPT-5.5 (DeerFlow).

![](images/5ec152ec37a10b142c763f8d2190f8ac6e343ef3a84c6459b6f48773490f4ea5.jpg)  
Figure S12 A failed Competitive Visual Disambiguation trajectory from Qwen3.5-397B-A17B (DeerFlow).

![](images/a3fddee718c342a20f6b24604c83af40b09aa0b7f3489b3147a4a39c03cec704.jpg)

![](images/82cbcb1f9f91db4c8b8a1e58456e3ceaf82c4334ddcc79edb2b5b3710c82fde0.jpg)  
Figure S13 A successful Visual-Structure-Guided Search trajectory from Kimi-K3 (ReAct).

![](images/f8f2c58d1ad9a44afe01ee12f498e733ea8257a1896c967f4ffad43b6880832d.jpg)  
Figure S14 A successful Visual-Structure-Guided Search trajectory from Qwen3.6-35B-A3B (ReAct).

![](images/4d263d5d4fe84b801545cfb31aa2162ee57c26e0a01a1ff34415656face4f072.jpg)  
Figure S15 A failed Visual-Structure-Guided Search trajectory from DeepSeek-V4.1-Flash (ReAct).

![](images/24b20978d3a526a181dde58852816f633eaf6692a1e287f13eb93e366f02bf2a.jpg)  
Figure S16 A failed Visual-Structure-Guided Search trajectory from Qwen3.8-27B (ReAct).

![](images/f2184fd1c24bcf9d61b711592a4b14bd475886174f1892e3d595143513b5f122.jpg)

![](images/128c5381afb0e1845fa157462138a401b70f34fea4d8e6ac7a01b0615ea4e1af.jpg)  
Figure S17 A failed Visual-Structure-Guided Search trajectory from GPT-5.5 (ReAct).