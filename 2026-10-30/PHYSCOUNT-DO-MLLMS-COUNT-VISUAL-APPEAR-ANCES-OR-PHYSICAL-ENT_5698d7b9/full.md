# PHYSCOUNT: DO MLLMS COUNT VISUAL APPEAR-ANCES OR PHYSICAL ENTITIES?

Lidong Lu<sup>1</sup>, Jiaxin Lv<sup>1</sup>, Liangrui Dong<sup>1</sup>, Xinyi Mao<sup>1</sup>, Lidong Zou<sup>1</sup>, Zhenquan Li<sup>1</sup>, Tong Lu<sup>1</sup>, Guo Chen<sup>2</sup>

<sup>1</sup>Nanjing University <sup>2</sup>NVIDIA

https://PhysCount.github.io

## ABSTRACT

Images reveal appearances, but counting often asks about physical objects. A single entity may be visible directly, in several reflections, or through glass, so detecting every visible occurrence does not determine how many entities are present. Existing visual counting benchmarks rarely separate these two operations. We introduce PhysCount, a benchmark for evaluating whether multimodal large language models (MLLMs) can distinguish what appears in an image from what physically exists in the scene. PhysCount contains 546 real-world images and 4,595 human-written questions organized into 651 target-specific groups. For each group, annotators locate the relevant appearances, label how each is observed, and identify appearances of the same entity. Four complementary question families test the enumeration of appearances, the enumeration of physical entities, imageformation-specific counting, and appearance–entity correspondence. We measure both accuracy on each family and the proportion of groups answered entirely correctly. Across 40 MLLMs, counting visible appearances is substantially easier than recovering the identities that connect them. Even the strongest model answers every question correctly in only 13.36% of groups. Performance declines when one entity produces many appearances or when its appearances cross observation modes. Higher image resolution improves access to visual evidence but brings limited gains in complete-group accuracy, whereas answering related questions jointly produces a marked improvement. These results expose a gap between recognizing what is visible and correctly recovering the physical entities behind those observations.

## 1 INTRODUCTION

Intelligent perception requires recovering a stable physical world from changing and sometimes misleading appearances. Developmental psychology studies this ability through the appearance– reality distinction: children gradually learn that changes in how an object looks do not imply changes in what the object is Flavell et al. (1983). Theories of object perception describe a similar process in which observations are organized around persistent physical units Spelke (1990), with “object files” connecting different appearances that are understood as states of the same object Kahneman et al. (1992). The importance of this ability becomes especially clear when visible evidence conflicts with physical structure. Mirror self-recognition experiments test whether an animal relates a reflected body to itself Gallup (1970), while a transparent barrier leaves a target fully visible but physically inaccessible Kabadayi et al. (2017). In both cases, successful behavior depends on interpreting the observation through the entities and relations that produce it. Sensory appearances provide evidence about the world, but they are not themselves the physical units into which the world is organized.

This distinction is equally consequential for multimodal large language models (MLLMs). A mirror can make one object appear multiple times in an image without creating additional entities, and an object seen through glass remains a physical entity despite the intervening surface. A model may therefore identify every visible instance and still answer incorrectly if it treats each appearance as a separate thing. Counting in such scenes requires more than locating targets and adding them up;

the model must determine which observations arise from the same entity. This leads to the central question of this work: Do MLLMs count visual appearances or physical entities?

![](images/7f07d91d006199cd371fbf3fc0a55fce70775c88d2f99430ca0de9048a24fd37.jpg)  
Figure 1: One scene, different counting units. Five cat appearances correspond to four entities because D1 and R1 depict the same cat. With the image and target fixed, the questions count all appearances (5), distinct entities (4), reflections (2), or entities seen both directly and in reflection (1).

Existing visual counting tasks typically begin with a target category and treat visible instances as the units to be counted, whether through question answering, category-agnostic counting, languageguided density estimation, or multimodal numerosity evaluation Acharya et al. (2019); Ranjan et al. (2021); Jiang et al. (2023); Paiss et al. (2023). These formulations evaluate whether a model can locate and enumerate the requested targets, while leaving the counting unit implicit. Mirrors, transparent surfaces, and repeated imaging paths separate visible appearances from physical entities. One entity may produce several spatially distinct observations, so the same visual inventory can support different counts depending on how those observations are grouped. Answering “how many” therefore requires both recovering the appearances and determining which of them share a physical source.

We introduce PhysCount, a benchmark for testing whether MLLMs can distinguish visible appearances from the physical entities they depict. Built from 546 natural images, PhysCount comprises 651 target-specific groups and 4,595 questions. For each counting target, annotators mark all relevant appearances, record whether each is seen directly, in reflection, or through glass, and group appearances that depict the same entity. These annotations generate four types of questions about appearances, physical entities, image formation, and correspondence. Questions associated with the same image and target probe complementary quantities derived from one annotated scene. PhysCount evaluates each count separately and reports how often a model solves the complete question group.

Experiments with 40 MLLMs reveal a clear separation between appearance counting and questions that depend on physical identity. Errors increase when the correct count requires grouping repeated appearances or matching observations across different modes. Scaling and higher resolution improve individual answers more readily than they improve complete-group accuracy. Presenting related questions together substantially improves both aggregate accuracy and the number of groups solved in full. Across these results, the largest remaining difficulty lies in counts defined by appearance– entity correspondence rather than in appearance enumeration alone.

Our contributions can be summarized as follows:

• We identify an overlooked distinction in visual counting: visible appearances and physical entities are not always in one-to-one correspondence, so a correct numerical answer does not necessarily show that a model has counted the intended units.

• We introduce PhysCount, comprising 546 natural images, 651 target-specific groups, and 4,595 questions. Its annotations record appearance locations, image-formation modes, and same-entity assignments, supporting questions about appearances, physical entities, image formation, and correspondence, together with strict complete-group evaluation.

• We evaluate 40 representative MLLMs and show that appearance counts are more reliable than counts requiring physical-identity correspondence. Performance declines on repeated and cross-mode appearances, while joint question presentation improves complete-group accuracy.

## 2 RELATED WORK

Visual Counting. Visual counting benchmarks have broadened both the form of supervision and the range of scenes under evaluation. TallyQA frames counting as question answering in natural images, while category-agnostic and language-guided methods study transfer to previously unseen targets Acharya et al. (2019); Ranjan et al. (2021); Jiang et al. (2023); Paiss et al. (2023). Recent multimodal benchmarks further examine dense and open-world counting, numerical question answering, and the counting behavior of general-purpose vision–language models Mondal et al. (2025); Deitke et al. (2025); Tamarapalli et al. (2025); Weng et al. (2025); Deng et al. (2026). Other work changes the information that must be selected or inferred: MIND and PairTally emphasize linguistic target specification Ciampi et al. (2025); Nguyen et al. (2025), CAPTURe recovers counts under occlusion Pothiraj et al. (2025), and video or audio-visual benchmarks extend enumeration across time and modalities Lu et al. (2026); Tsuchiya et al. (2026); Rong et al. (2026). This progression has made counting more open-ended, compositional, and robust to incomplete evidence, but the unit being counted is still generally a visible category instance or an instance inferred to occupy the scene. Work on numerical cognition offers a complementary view: small-set judgments and approximate numerosity are distinct processes, while object-based accounts tie number to representations that persist across changes in view Mandler & Shebo (1982); Feigenson et al. (2004); Wynn (1992); Kahneman et al. (1992). PhysCount extends this distinction to MLLMs by evaluating two abilities separately: enumerating visible appearances and grouping appearances that depict the same physical entity.

Mirror Understanding in MLLMs. Recent benchmarks examine mirror understanding in MLLMs from two distinct perspectives. MirrorQA evaluates subject-centered left–right reasoning in mirror images, testing whether models correctly transform spatial orientation Liu et al. (2026). MirrorBench adopts an embodied perspective: inspired by the mirror self-recognition paradigm, it evaluates self-centric intelligence from basic mirror perception to higher-level self-representation in simulated environments Guo et al. (2026). PhysCount targets a different capability. Rather than asking models to infer orientation or recognize themselves, it asks them to enumerate the target appearances in a scene and determine which of those appearances depict the same physical entity. This requires handling mirror-only entities, direct views paired with reflections, and multiple reflections generated by one entity. PhysCount represents these cases explicitly by annotating each appearance, its image-formation mode, and its physical-identity group, then evaluates the resulting appearance, entity, and correspondence counts.

## 3 PHYSCOUNT: COUNTING BEYOND VISIBLE APPEARANCES

## 3.1 TASK FORMULATION

PhysCount presents a model with an image and a target-specific counting question and expects an integer answer. The number of visible target appearances may differ from the number of objects in the scene. The same object may be seen directly and in one or more mirrors, while another may be visible only through reflection or glass. The question determines whether the model should count each visible appearance or merge appearances that depict the same physical entity.

PhysCount evaluates four question families: counting all visible appearances, counting physical entities, counting a specified type of appearance, and reasoning about appearance–entity correspondences. The same image can consequently admit several correct counts depending on what the question specifies. Together, these questions separate the recovery of visual evidence from the interpretation of that evidence in terms of physical entities.

## 3.2 DATA COLLECTION AND ANNOTATION

Figure 2 summarizes the construction of PhysCount in three stages. First, we build the image set through automatic candidate filtering followed by human selection. Qwen3.6-35B-A3B Qwen Team (2026b) filters natural images from Common Canvas Gokaslan et al. (2024), GPIC Chandrasegaran et al. (2026), and other public collections for mirror and glass scenes in which reflections or glass change the apparent count. Human annotators then inspect these candidates and select images with localizable targets, distinguishable direct, reflected, or through-glass appearances, and counts that can be resolved through careful inspection. The selected images contain salient appearance duplica tion or interactions between reflected and transmitted content.

![](images/c758b9d3274a48e9d1ee852e12e1a36055d2e08718367c38d11c83d61810e743.jpg)  
Figure 2: PhysCount construction pipeline. Candidate mirror and glass scenes undergo automatic filtering and human selection. An annotator labels direct, reflected, and through-glass appearances, groups appearances of the same entity, and writes questions whose answers are derived from these groups. Here, five cat appearances form four entities, and the correspondence question selects the one entity seen both directly and in reflection. Annotation review and automated QA checking send consistent QAs to PhysCount and flagged QAs to final human correction.

Second, annotators construct a target-specific scene representation. An image containing several suitable target categories produces a separate annotation group for each category. Within a group, annotators mark every visible target occurrence and label it as direct, reflected, or seen through glass. Occurrences depicting the same real-world object receive a common identity. Thus, each point records the location and type of one appearance, while each identity group records the physical entity depicted by one or more appearances. The grouping covers direct–reflection pairs, objects visible only in mirrors, and multiple reflections of one object.

Finally, the first annotator writes counting questions and their integer answers based on the completed scene annotation. A second annotator checks the appearance locations, appearance-type labels, and identity groups against the image. GPT-5.6 Sol then uses the verified annotation to check whether each human-written question can be answered from the recorded information and whether its answer gives the required count. A third annotator reviews and corrects every flagged QA pair. Appendix A provides the complete instructions and examples.

## 3.3 QUESTION DESIGN

PhysCount uses four question families to expose different failure points between seeing a target and recovering the physical scene. The image and target remain fixed while the required count changes, so success depends on selecting the correct counting unit and applying the required operation to the same annotated evidence. Figure 1 shows how the same scene yields different answers as the requested counting unit changes.

Appearance questions ask for every visible occurrence of the target, including direct, reflected, and through-glass appearances. The difficulty is to identify separate occurrences before deciding whether they share an identity. Reflections may be cropped, distorted, or repeated across mirrors, while reflected and transmitted content may overlap on the same glass surface. A model that merges similar-looking regions too early will miss valid appearances; one that splits a single occurrence into fragments will overcount them. Each annotation group includes one appearance question to measure this initial visual inventory.

Entity questions ask how many physical objects are represented by that inventory. This requires matching appearances despite changes in position, scale, orientation, visibility, and appearance type. A direct observation and several mirror images may all depict one object, whereas two similar reflections may belong to different objects. Mirror-only entities add a further challenge because identity must be established without a direct appearance as an anchor. The resulting errors have two opposite forms: counting one object several times or merging distinct objects into one.

Image-formation questions ask whether each appearance is seen directly, formed by reflection, or observed through glass. Object recognition alone does not determine this label: the same person or object can be visible directly, in reflection, and through glass in one image. Glass is especially demanding because reflected foreground content and transmitted background content occupy the same pixels, and their boundaries may cross or partly hide one another. The model must separate these layers and determine how each target occurrence becomes visible.

Correspondence questions examine whether the recovered appearances, appearance types, and identities agree. They count intersections, exclusive sets, and multiplicities, such as entities seen both directly and in reflection or entities producing several mirror images. These answers depend on the full pattern of identity assignments: changing one appearance–entity match can alter several relation counts at once. The challenge is therefore not a single local match, but maintaining one consistent set of identities across all observed appearances.

## 3.4 BENCHMARK STATISTICS

PhysCount contains 546 images, 651 target-specific annotation groups, and 4,595 factual questions covering 137 normalized counting targets. Every group includes an appearance question, and 599 groups include all four question families described above. A group contains 7.06 questions on average. As shown in Figure 3(b), correspondence is the largest family, with 2,083 questions: 924 concern visibility sets, 459 ask about overlap among direct, reflected, and through-glass appearances, and 700 examine how many appearances are associated with each entity.

The annotations contain 7,086 target appearances, including 2,416 direct appearances, 3,996 reflections, and 674 appearances seen through glass. Among the 651 groups, 475 include an entity visible through reflection or glass with no direct appearance. Figure 3(a) relates the number of annotated appearances to the number of physical entities. The 229 groups on y = x contain one appearance per entity; 166 contain indirect appearances alone, while 63 combine direct and indirect appearances belonging to different entities. The other 422 groups contain at least one entity with multiple appearances. In 402 groups, the same entity has appearances of different types, such as a direct appearance and a reflection. In the other 20, the same entity appears several times, but all of its appearances have the same type, as when one object produces several reflections.

Figure 3(c) shows the diversity and long tail of the 137 counting targets. They include people, animals, vehicles, furniture, containers, lighting, apparel, food, plants, toys, architectural elements, and many less frequent everyday objects. Answer values range from 0 to 196 and differ across question families: appearance questions have a mean answer of 10.88, while correspondence questions usually count smaller sets defined by identity relations.

![](images/57068f01fa51bf21696ca8a44109932950cefd4f0baa8a978e0031f521f0d514.jpg)  
Figure 3: Benchmark statistics. (a) Visual-appearance and physical-entity counts across 651 annotation groups; bubble area encodes the number of groups at each coordinate, and the adjacent chart shows the appearance-type composition of the 229 groups on y = x. (b) Distribution of 4,595 questions across the four question families. (c) Diversity and frequency of counting targets.

Table 1: Results on PhysCount for all 40 evaluated models and five human participants. Each participant independently answers all 4,595 questions under the same image–question protocol as the models; the human row reports their mean. Best and second-best model results are bolded and underlined.
<table><tr><td></td><td colspan="2">Appearance</td><td colspan="2">Entity</td><td colspan="2">Image Formation</td><td colspan="2">Correspondence</td><td colspan="3"></td></tr><tr><td>Model</td><td>MAE↓</td><td>NAE↓</td><td>MAE↓</td><td>NAE↓</td><td>MAE↓</td><td>NAE↓</td><td>MAE↓</td><td>NAE↓</td><td>Score↑ Group-Acc↑ Phys-Gnd↑</td><td></td><td></td></tr><tr><td>Human (5-person avg.)</td><td>0.037</td><td>0.59</td><td>0.023</td><td>0.83</td><td>0.035</td><td>1.05</td><td>0.033</td><td>2.21</td><td>97.10</td><td>95.98</td><td>99.49</td></tr><tr><td>Gemini 3.7 Flash Google DeepMind (2026)</td><td>2.724</td><td>20.08</td><td>2.046</td><td>27.36</td><td>2.467</td><td>30.62</td><td>1.396</td><td>33.88</td><td>36.45</td><td>13.36</td><td>73.00</td></tr><tr><td>GPT-5.6 Sol OpenAI (2026a)</td><td>3.028</td><td>22.82</td><td>2.808</td><td>36.58</td><td>2.544</td><td>31.94</td><td>2.113</td><td>52.11</td><td>21.73</td><td>3.84</td><td>55.65</td></tr><tr><td>GPT-6 Astra OpenAI (2026b)</td><td>3.049</td><td>24.87</td><td>2.673</td><td>37.72</td><td>2.517</td><td>30.99</td><td>1.905</td><td>47.83</td><td>20.77</td><td>3.38</td><td>51.89</td></tr><tr><td>GPT-5.6 Terra OpenAI (2026a)</td><td>3.395</td><td>26.06</td><td>2.726</td><td>36.39</td><td>3.227</td><td>37.30</td><td>1.863</td><td>50.99</td><td>20.42</td><td>4.30</td><td>53.43</td></tr><tr><td>GPT-5.6 Luna OpenAI (2026a)</td><td>3.536</td><td>28.24</td><td>3.368</td><td>42.24</td><td>2.992</td><td>35.07</td><td>2.283</td><td>55.98</td><td>16.87</td><td>2.92</td><td>46.98</td></tr><tr><td>Qwen3.8-27B Qwen Team (2026c)</td><td>3.149</td><td>19.89</td><td>2.511</td><td>32.42</td><td>2.702</td><td>29.66</td><td>1.685</td><td>44.68</td><td>28.84</td><td>7.37</td><td>65.96</td></tr><tr><td>Qwen3.5-397B-A17B Qwen Team (2026a)</td><td>2.836</td><td>18.99</td><td>2.442</td><td>33.28</td><td>2.386</td><td>26.88</td><td>1.859</td><td>46.73</td><td>27.95</td><td>5.84</td><td>62.16</td></tr><tr><td>Qwen3.5-122B-A10B Qwen Team (2026a)</td><td>4.192</td><td>17.96</td><td>2.527</td><td>34.40</td><td>2.603</td><td>27.80</td><td>2.068</td><td>49.00</td><td>26.03</td><td>5.99</td><td>57.13</td></tr><tr><td>Qwen3.5-35B-A3B Qwen Team (2026a)</td><td>3.143</td><td>18.81</td><td>2.425</td><td>33.44</td><td>2.621</td><td>28.49</td><td>2.011</td><td>51.51</td><td>25.70</td><td>5.99</td><td>58.76</td></tr><tr><td>Qwen3.5-27B Qwen Team (2026a)</td><td>4.103</td><td>20.64</td><td>3.216</td><td>34.58</td><td>3.017</td><td>26.78</td><td>2.942</td><td>48.56</td><td>26.32</td><td>6.30</td><td>60.83</td></tr><tr><td>Qwen3.5-9B Qwen Team (2026a)</td><td>2.845</td><td>20.04</td><td>3.708</td><td>36.09</td><td>3.805</td><td>30.05</td><td>4.816</td><td>53.38</td><td>23.54</td><td>5.07</td><td>55.41</td></tr><tr><td>Qwen3.5-4B Qwen Team (2026a)</td><td>3.321</td><td>22.19</td><td>4.379</td><td>39.57</td><td>2.729</td><td>32.93</td><td>3.274</td><td>57.03</td><td>18.57</td><td>3.38</td><td>48.20</td></tr><tr><td>Gemma 4-31B Team (2026a)</td><td>3.307</td><td>21.83</td><td>2.700</td><td>36.88</td><td>2.814</td><td>33.71</td><td>1.879</td><td>41.86</td><td>26.32</td><td>6.61</td><td>60.07</td></tr><tr><td>Gemma 4-26B-A4B Team (2026a)</td><td>3.257</td><td>22.70</td><td>2.852</td><td>39.22</td><td>2.936</td><td>36.53</td><td>1.888</td><td>44.02</td><td>23.79</td><td>5.53</td><td>57.17</td></tr><tr><td>Gemma 4-12B Team (2026a)</td><td>3.490</td><td>26.05</td><td>3.044</td><td>44.56</td><td>3.057</td><td>37.08</td><td>1.982</td><td>45.70</td><td>19.03</td><td>3.07</td><td>46.10</td></tr><tr><td>Gemma 4-E4B Team (2026a)</td><td>37.605</td><td>50.27</td><td>22.506</td><td>64.37</td><td>22.469</td><td>69.95</td><td>9.344</td><td>65.25</td><td>2.93</td><td>0.00</td><td>11.90</td></tr><tr><td>Gemma 4-E2B Team (2026a)</td><td>20.343</td><td>48.02</td><td>8.357</td><td>55.57</td><td>11.088</td><td>55.82</td><td>4.146</td><td>62.93</td><td>4.76</td><td>0.15</td><td>21.13</td></tr><tr><td>InternVL3.5-241B-A28B Wang et al. (2025)</td><td>4.135</td><td>30.21</td><td>2.825</td><td>36.24</td><td>4.597</td><td>36.63</td><td>1.983</td><td>48.11</td><td>21.04</td><td>4.92</td><td>60.27</td></tr><tr><td>InternVL3.5-38B Wang et al. (2025)</td><td>4.078</td><td>32.36</td><td>4.520</td><td>39.75</td><td>10.133</td><td>36.59</td><td>3.837</td><td>57.92</td><td>15.55</td><td>2.61</td><td>47.23</td></tr><tr><td>InternVL3.5-30B-A3B Wang et al. (2025)</td><td>4.152</td><td>29.51</td><td>3.036</td><td>38.94</td><td>3.359</td><td>37.29</td><td>2.108</td><td>52.08</td><td>17.76</td><td>2.61</td><td>53.91</td></tr><tr><td>InternVL3.5-14B Wang et al. (2025)</td><td>4.158</td><td>29.13</td><td>2.855</td><td>40.02</td><td>3.483</td><td>38.22</td><td>2.099</td><td>57.32</td><td>15.20</td><td>1.84</td><td>44.22</td></tr><tr><td>InternVL3.5-8B Wang et al. (2025)</td><td>3.938</td><td>27.82</td><td>3.011</td><td>41.68</td><td>3.670</td><td>42.16</td><td>2.233</td><td>57.78</td><td>14.29</td><td>1.84</td><td>47.08</td></tr><tr><td>InternVL3.5-4B Wang et al. (2025)</td><td>7.018</td><td>38.11</td><td>3.452</td><td>48.83</td><td>3.685</td><td>39.85</td><td>2.773</td><td>65.97</td><td>9.53</td><td>1.23</td><td>36.93</td></tr><tr><td>Muse-Glimmer-30B Meta Superintelligence Labs (2026)</td><td>2.943</td><td>20.31</td><td>2.478</td><td>32.10</td><td>2.405</td><td>28.69</td><td>1.608</td><td>43.04</td><td>29.61</td><td>7.83</td><td>65.51</td></tr><tr><td>DeepSeek-V4-Flash-Vision DeepSeek-AI (2026)</td><td>4.321</td><td>30.52</td><td>2.814</td><td>36.12</td><td>3.406</td><td>37.19</td><td>1.864</td><td>48.56</td><td>20.74</td><td>5.22</td><td>61.61</td></tr><tr><td>MiniMax-M3-428B-A23B Lai et al. (2026)</td><td>3.888</td><td>29.28</td><td>2.989</td><td>39.05</td><td>3.739</td><td>39.57</td><td>1.975</td><td>52.65</td><td>17.43</td><td>3.69</td><td>52.50</td></tr><tr><td>MiniCPM-V 4.5-8B yu2 (2025)</td><td>4.528</td><td>32.52</td><td>2.828</td><td>38.82</td><td>3.426</td><td>39.32</td><td>2.176</td><td>58.40</td><td>14.90</td><td>2.76</td><td>51.28</td></tr><tr><td>MiniCPM-V 4.6-1.3B Yao et al. (2024)</td><td>5.720</td><td>39.90</td><td>3.340</td><td>43.56</td><td>3.694</td><td>42.36</td><td>2.252</td><td>57.22</td><td>12.40</td><td>2.15</td><td>45.11</td></tr><tr><td>Cosmos3-Super-64B NVIDIA et al. (2026)</td><td>4.144</td><td>28.29</td><td>3.003</td><td>37.39</td><td>3.051</td><td>33.50</td><td>2.339</td><td>64.08</td><td>14.69</td><td>1.38</td><td>45.67</td></tr><tr><td>Cosmos3-Nano-16B NVIDIA et al. (2026)</td><td>6.002</td><td>30.81</td><td>3.239</td><td>41.23</td><td>3.311</td><td>36.84</td><td>2.548</td><td>68.01</td><td>12.08</td><td>2.15</td><td>36.23</td></tr><tr><td>Cosmos3-Edge-4B NVIDIA et al. (2026)</td><td>5.332</td><td>37.23</td><td>4.005</td><td>43.59</td><td>3.555</td><td>41.68</td><td>3.209</td><td>65.98</td><td>9.27</td><td>0.77</td><td>32.42</td></tr><tr><td>Ovis2.6-80B-A3B Lu et al. (2024) Ovis2.6-30B-A3B Lu et al. (2024)</td><td>4.132</td><td>31.39</td><td>3.089</td><td>39.59</td><td>3.313</td><td>37.88</td><td>2.302</td><td>58.70</td><td>14.93</td><td>2.15</td><td>46.79</td></table>

## 3.5 EVALUATION PROTOCOL

For a non-negative integer prediction $p$ and ground-truth count $y ,$ we report absolute error and bounded normalized absolute error,

$$
\mathrm { A E } ( p , y ) = | p - y | , \qquad \mathrm { N A E } ( p , y ) = \mathrm { m i n } \biggl ( 1 , { \frac { | p - y | } { \mathrm { m a x } ( y , 1 ) } } \biggr ) .\tag{1}
$$

Outputs that cannot be parsed as valid counts receive the maximum normalized error, while MAE is computed over parsed predictions. For each question family, we first average the NAE of that family’s questions within each annotation group. We then average these group-level values over all groups containing that family. A group therefore contributes one value regardless of how many questions of that family it contains. Let $a _ { g } ^ { t }$ denote one minus the within-group NAE for family t, and let $\mathcal { T } _ { g }$ contain the families available in group g. The primary aggregate metric is

$$
\mathrm { S c o r e } = \frac { 1 0 0 } { | \mathcal { G } | } \sum _ { g \in \mathcal { G } } \prod _ { t \in \mathcal { T } _ { g } } a _ { g } ^ { t } .\tag{2}
$$

The product rewards reliable performance across all question families in a group: strong appearance enumeration cannot offset a failed entity or correspondence estimate. We additionally report Group-Acc, the percentage of annotation groups for which every answer exactly matches its ground truth. It measures complete group-level correctness. Phys-Gnd isolates physical reasoning after successful enumeration. Among groups covering all four question families, it retains those whose total-appearance answer is exact, computes the harmonic mean of entity, image-formation, and correspondence accuracy, and then averages across the retained groups. An eligible group receives zero if any of these three components is zero. Because the retained groups depend on which appearance questions a model answers correctly, Phys-Gnd measures conditional physical grounding rather than overall capability.

## 4 EXPERIMENTS

## 4.1 EVALUATION SETTING

We evaluate 40 proprietary and open-weight MLLMs under a common task protocol. Unless stated otherwise, each question is answered independently from the same image, preventing related questions in an annotation group from sharing previous answers. Each model uses the strongest reasoning or thinking setting available in its evaluated backend and returns a single integer. Five human participants independently complete all 4,595 questions using the same image–question inputs; their metrics are averaged to form the human baseline. We preserve the native image resolution up to a 4,096-pixel long edge and proportionally downsample larger images. The resolution and jointquestion studies retain the same model configuration and image processing. Appendix B provides the prompts, backend-specific inference settings, and output-validation rules.

## 4.2 MAIN RESULTS

Table 1 shows that Gemini 3.7 Flash achieves the highest overall Score, while Muse-Glimmer-30B and Qwen3.8-27B are the strongest open models. No model leads every metric. Qwen3.5- 397B-A17B has the lowest appearance error and performs strongly on image formation, whereas Gemma 4-31B is more competitive on correspondence. Neither advantage produces the best overall Score. In contrast, the five-participant human average reaches 97.10 Score and 95.98 Group-Acc, showing that careful human observers can resolve nearly all quantities in a group together. The results distinguish three related capabilities: detecting what is visible, recognizing how each appearance is formed, and recovering the physical entities behind those appearances. Strength in one does not ensure strength in the others.

Across models, enumerating visible appearances is consistently easier than answering entity and correspondence questions. These questions use the same scene and counting target, but the latter require deciding which direct, reflected, and through-glass appearances depict the same object. Phys-Gnd remains well below its ceiling even on groups with an exact appearance answer. A correct appearance total therefore does not reliably carry over to counts defined by physical identity. The benchmark identifies this separation at the level of answer behavior; integer outputs alone do not determine which internal computation produced an error.

Group-Acc sets a stricter requirement: every appearance, entity, image-formation, and correspondence question in a group must be answered exactly. Many models answer a substantial fraction of individual questions correctly, yet only Gemini solves more than 10% of annotation groups in full. Stronger model variants generally reduce error within each question family, but their Group-Acc remains low. Increased model capacity therefore improves individual counts more readily than it produces a complete solution to all quantities derived from the same annotated scene.

## 4.3 DETAILED ANALYSIS

Impact of Input Resolution. Table 2 traces how additional visual detail affects the counting process. Increasing the long-edge resolution improves the overall Score from 21.04 to 28.84, but most gains occur before the highest resolution. Appearance and correspondence errors reach their lowest values at 2,048 pixels and change little thereafter, while Group-Acc stops improving after 1,024 pixels. The different saturation points separate visual access from complete task success. Higher resolution helps models recover visible appearances, whereas solving an entire group also requires exact answers to its entity, image-formation, and correspondence questions. Once the visible inven tory stabilizes, further pixels contribute little to these identity-dependent counts.

Table 2: Impact of input resolution: Qwen3.8-27B under independent questioning at four maximum long-edge resolutions.
<table><tr><td>Metric</td><td>512</td><td>1,024</td><td>2,048</td><td>4,096</td></tr><tr><td>Appearance NAE↓</td><td>28.36</td><td>21.53</td><td>19.42</td><td>19.89</td></tr><tr><td>Entity NAE↓</td><td>37.95</td><td>33.88</td><td>33.09</td><td>32.42</td></tr><tr><td>Formation NAE↓</td><td>37.08</td><td>32.12</td><td>31.69</td><td>29.66</td></tr><tr><td>Correspond. NAE↓</td><td>49.31</td><td>46.85</td><td>44.22</td><td>44.68</td></tr><tr><td>Score↑</td><td>21.04</td><td>26.41</td><td>27.86</td><td>28.84</td></tr><tr><td>Group-Acc↑</td><td>5.38</td><td>7.37</td><td>7.37</td><td>7.37</td></tr><tr><td>Phys-Gnd↑</td><td>62.50</td><td>63.10</td><td>61.26</td><td>65.96</td></tr></table>

Table 3: Impact of question presentation: Qwen3.8-27B at a 4,096-pixel long edge with questions presented independently or jointly by annotation group.  
![](images/e6b74dbf53b9751c3f2784190e713002d5a41ea0c228c8f933ba0360f3c4ee67.jpg)

<table><tr><td>Metric</td><td>Independent</td><td>Joint</td></tr><tr><td>Appearance NAE↓</td><td>19.89</td><td>19.37</td></tr><tr><td>Entity NAE↓</td><td>32.42</td><td>26.26</td></tr><tr><td>Formation NAE↓</td><td>29.66</td><td>32.76</td></tr><tr><td>Correspond. NAE↓</td><td>44.68</td><td>33.39</td></tr><tr><td>Score↑</td><td>28.84</td><td>40.17</td></tr><tr><td>Group-Acc↑</td><td>7.37</td><td>23.96</td></tr><tr><td>Phys-Gnd↑</td><td>65.96</td><td>78.56</td></tr></table>

Figure 4: Target-domain performance: group-macro Score of four representative models across 16 mutually exclusive counting-target domains.

Impact of Question Presentation. Table 3 compares independent and joint presentation. Joint presentation raises Score by more than eleven points and more than triples Group-Acc, with gains concentrated in entity and correspondence questions despite slightly worse image-formation performance. Questions in the same annotation group refer to the same appearances and physical entities but request different counts. Seeing these questions together gives the model access to the full set of counting operations associated with the scene. The large reductions in entity and correspondence error show that this presentation is especially helpful for identity-dependent counts, while the rise in Group-Acc means that substantially more annotation groups are solved in full.

Performance across Target Domains. Figure 4 reveals clear differences in how stable model performance remains across counting targets. Gemini 3.7 Flash follows a relatively even profile, avoiding pronounced drops in several categories without leading every domain. Qwen3.8-27B and Gemma 4-31B reach higher peaks on selected targets but vary more across categories, and their local strengths do not transfer consistently to other targets. Several curves nevertheless rise and fall together across domains, showing that the target category systematically changes the difficulty of separating appearances and matching them to entities. The model-specific peaks further indicate that models rely on different category knowledge and identity cues.

![](images/193924c245050b0f17000c0e5c78741aba0d7dd117d9c7302be519390f8297a6.jpg)

![](images/75a74806d39987b49111a02005d1271139a0d6f5115598d03cb69a9d48c91ec0.jpg)  
Figure 5: Performance by appearance multiplicity. Entity NAE (a) and Score (b) grouped by the maximum number of appearances assigned to one entity; dashed lines show the equal-weight mean across all 40 models.  
Figure 6: Within- v.s. cross-mode correspondence. Correspondence NAE (a) and Score (b) after matching the appearance-multiplicity distributions of the two group types; annotations report the cross-mode minus within-mode difference.

Impact of Appearance Multiplicity. Figure 5 shows that appearance multiplicity affects entity counting and joint performance at different stages. The 40-model mean entity NAE changes little from one to three appearances, then rises sharply in the four-or-more bin. Mean Score declines as multiplicity increases from one to two, partially recovers at three while remaining below the singleappearance level, and reaches its minimum at four or more. Gemini 3.7 Flash declines steadily across the four bins, whereas the other representative models fluctuate at intermediate multiplicities; nevertheless, 32 of the 40 models score lower in the highest-multiplicity bin than in the single-appearance bin. A small number of repeated appearances can therefore leave the entity count largely intact while already reducing the likelihood that appearance, entity, image-formation, and correspondence questions are all answered reliably. Each additional appearance supplies more visual evidence but also adds another correspondence that the requested entity count must resolve. These effects can offset each other at lower multiplicities; with four or more appearances, both entity counting and joint performance deteriorate.

Impact of Matching across Observation Modes. After matching the appearance-multiplicity distributions, Figure 6 shows higher cross-mode correspondence error for all four representative models, with the same pattern appearing in 31 of the 40 evaluated models. Counts based on identity correspondence are thus sensitive not only to the number of observations, but also to how those observations are formed. Models can answer simpler appearance questions correctly while still missing counts that require direct, reflected, and transmitted observations to be treated as views of the same entity. Aggregate Score does not decline in parallel because direct views can simplify individual counts and offset errors in cross-mode correspondence. A competitive Score can therefore coexist with substantial errors on questions that require cross-mode identity matching.

Discussion. Across the four analyses, models improve at enumerating appearances before they reliably answer counts defined by physical identity. Higher resolution improves access to visible evidence, while repeated and cross-mode observations remain difficult; the joint-prompting experiment also improves complete-group correctness. The central challenge is therefore to organize observations around persistent physical entities rather than count them in isolation. Future evaluation can complement final counts with explicit appearance and identity predictions, separating missed evidence from incorrect grouping. For physical AI, this points toward scene understanding whose entity-level predictions remain stable across image formation, viewpoint, and time NVIDIA et al. (2026).

## 5 CONCLUSION

We introduced PhysCount, a benchmark for evaluating whether MLLMs distinguish visual appearances from the physical entities that produce them. By deriving 4,595 related questions from shared appearance–entity annotations, PhysCount evaluates individual counts and complete-group correctness across appearance, entity, image-formation, and correspondence questions. Experiments with 40 MLLMs show that appearance counts are more reliable than counts requiring physical-identity correspondence, particularly when entities appear repeatedly or across observation modes. Higher resolution reduces appearance error without eliminating the remaining identity-dependent errors, while joint presentation substantially increases the number of groups solved in full. These results separate two observable capabilities: enumerating visible evidence and answering counts defined by how that evidence maps to physical objects. Progress toward physical scene understanding should make entity-centered predictions stable across reflection, transmission, viewpoint, and time. Such stability would support reasoning about enduring objects and their interactions rather than treating each visual occurrence as an independent counting unit. PhysCount exposes the gap between counting what is visible and correctly counting the physical entities those observations depict.

## AI USE STATEMENT

Generative AI tools assisted candidate-image screening, consistency checks of question–answer pairs against structured annotations, language editing, and manuscript organization. Human annotators made the final image-selection decisions, created and reviewed the annotations and questions, and adjudicated cases flagged during QA checking. The authors reviewed the resulting data, text, claims, and artifacts and take full responsibility for the work.

## REPRODUCIBILITY STATEMENT

Section 3.5 defines the metrics and aggregation procedure, and Section 4.1 specifies the common evaluation setting. Appendix A provides the scene-selection criteria, screening and QA-audit prompts, annotation-verification procedure, and representative examples pairing questions and answers with their structured annotations. Appendix B gives the evaluation prompts, image preprocessing, inference settings, human-baseline protocol, and answer-validation rules. Appendix C reports the complete numerical results underlying the diagnostic analyses. The complete PhysCount benchmark will be released upon acceptance.

## REFERENCES

MiniCPM-V4.5: Cooking Efficient MLLMs via Architecture, Data, and Training Recipe, 2025. URL https://arxiv.org/abs/2509.18154.

Manoj Acharya, Kushal Kafle, and Christopher Kanan. Tallyqa: Answering complex counting questions. Proceedings of the AAAI Conference on Artificial Intelligence, 33(01):8076–8084, July 2019. ISSN 2159-5399. doi: 10.1609/aaai.v33i01.33018076. URL http://dx.doi.org/10. 1609/aaai.v33i01.33018076.

Xiang An, Yin Xie, Feilong Tang, Yunyao Yan, Huajie Tan, Didi Zhu, Changrui Chen, Xiuwei Zhao, Bin Qin, Kaicheng Yang, Yifei Shen, Yuanhan Zhang, Kaichen Zhang, Wenkang Zhang, Zheng Cheng, Nansen Zhang, Chunsheng Wu, Chunjiang Ge, Zimin Ran, Dehua Song, Chunyuan Li, Shikun Feng, Ming Hu, Zhangquan Chen, Junbo Niu, Bo Li, Ziyong Feng, Ziwei Liu, Zongyuan Ge, and Jiankang Deng. Llava-onevision-2: Towards next-generation perceptual intelligence, 2026. URL https://arxiv.org/abs/2605.25979.

Jyoti Aneja, Michael Harrison, Neel Joshi, Tyler LaBonte, John Langford, and Eduardo Salinas. Phi-4-reasoning-vision-15b technical report, 2026. URL https://arxiv.org/abs/2603.03975.

Baidu-ERNIE-Team. Ernie 4.5 technical report. https://ernie.baidu.com/blog/publication/ERNIE Technical Report.pdf, 2025.

Keshigeyan Chandrasegaran, Kyle Sargent, Suchir Agarwal, Michael Jang, Michael Poli, Juan Carlos Niebles, Justin Johnson, Jiajun Wu, and Li Fei-Fei. Gpic: A giant permissive image corpus for visual generation, 2026. URL https://arxiv.org/abs/2605.30341.

Luca Ciampi, Nicola Messina, Matteo Pierucci, Giuseppe Amato, Marco Avvenuti, and Fabrizio Falchi. Mind the prompt: A novel benchmark for prompt-based class-agnostic counting. In Proceedings of the Winter Conference on Applications of Computer Vision (WACV), pp. 7959– 7968, February 2025.

Christopher Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park, Rohun Tripathi, Sangho Lee, Mohammadreza Salehi, Jason Ren, Chris Dongjoo Kim, Yinuo Yang, Vincent Shao, Yue Yang, Weikai Huang, Ziqi Gao, Taira Anderson, Jianrui Zhang, Jitesh Jain, George Stoica, Ali Farhadi, and

Ranjay Krishna. Molmo2: Open weights and data for vision-language models with video understanding and grounding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 28652–28668, June 2026.

DeepSeek-AI. DeepSeek-V4-Flash-Vision: Vision input release notes, 2026. URL https://api-docs. deepseek.com/news/news260821/.

Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, Jiasen Lu, Taira Anderson, Erin Bransom, Kiana Ehsani, Huong Ngo, YenSung Chen, Ajay Patel, Mark Yatskar, Chris Callison-Burch, Andrew Head, Rose Hendrix, Favyen Bastani, Eli VanderBilt, Nathan Lambert, Yvonne Chou, Arnavi Chheda, Jenna Sparks, Sam Skjonsberg, Michael Schmitz, Aaron Sarnat, Byron Bischoff, Pete Walsh, Chris Newell, Piper Wolters, Tanmay Gupta, Kuo-Hao Zeng, Jon Borchardt, Dirk Groeneveld, Crystal Nam, Sophie Lebrecht, Caitlin Wittlif, Carissa Schoenick, Oscar Michel, Ranjay Krishna, Luca Weihs, Noah A. Smith, Hannaneh Hajishirzi, Ross Girshick, Ali Farhadi, and Aniruddha Kembhavi. Molmo and pixmo: Open weights and open data for state-ofthe-art vision-language models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 91–104, June 2025.

Jinhong Deng, Limeng Qiao, and Guanglu Wan. Holocount: A holistic visual counting benchmark for mllms, 2026. URL https://arxiv.org/abs/2607.06420.

Lisa Feigenson, Stanislas Dehaene, and Elizabeth Spelke. Core systems of number. Trends in Cognitive Sciences, 8(7):307–314, July 2004. ISSN 1364-6613. doi: 10.1016/j.tics.2004.05.002. URL http://dx.doi.org/10.1016/j.tics.2004.05.002.

John H. Flavell, Eleanor R. Flavell, and Frances L. Green. Development of the appearance-reality distinction. Cognitive Psychology, 15(1):95–120, January 1983. ISSN 0010-0285. doi: 10.1016/ 0010-0285(83)90005-1. URL http://dx.doi.org/10.1016/0010-0285(83)90005-1.

Gordon G. Gallup. Chimpanzees: Self-recognition. Science, 167(3914):86–87, January 1970. ISSN 1095-9203. doi: 10.1126/science.167.3914.86. URL http://dx.doi.org/10.1126/science.167.3914. 86.

Aaron Gokaslan, A. Feder Cooper, Jasmine Collins, Landan Seguin, Austin Jacobson, Mihir Patel, Jonathan Frankle, Cory Stephenson, and Volodymyr Kuleshov. Commoncanvas: Open diffusion models trained on creative-commons images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 8250–8260, June 2024.

Google DeepMind. Gemini 3.7 flash model card, August 2026. URL https://deepmind.google/ models/model-cards/gemini-3-7-flash/.

Shengyu Guo, Tongrui Ye, Jianbo Zhang, Zicheng Zhang, Chunyi Li, and Guangtao Zhai. Mirrorbench: Evaluating self-centric intelligence in mllms by introducing a mirror, 2026. URL https://arxiv.org/abs/2604.14785.

Ruixiang Jiang, Lingbo Liu, and Changwen Chen. Clip-count: Towards text-guided zero-shot object counting. In Proceedings of the 31st ACM International Conference on Multimedia, MM ’23, pp. 4535–4545. ACM, October 2023. doi: 10.1145/3581783.3611789. URL http://dx.doi.org/10. 1145/3581783.3611789.

Can Kabadayi, Katarzyna Bobrowicz, and Mathias Osvath. The detour paradigm in animal cognition. Animal Cognition, 21(1):21–35, December 2017. ISSN 1435-9456. doi: 10.1007/ s10071-017-1152-0. URL http://dx.doi.org/10.1007/s10071-017-1152-0.

Daniel Kahneman, Anne Treisman, and Brian J Gibbs. The reviewing of object files: Object-specific integration of information. Cognitive Psychology, 24(2):175–219, April 1992. ISSN 0010-0285. doi: 10.1016/0010-0285(92)90007-o. URL http://dx.doi.org/10.1016/0010-0285(92)90007-O.

Xunhao Lai, Weiqi Xu, Yufeng Yang, Qiaorui Chen, Yang Xu, Lunbin Zeng, Xiaolong Li, Haohai Sun, Haichao Zhu, Vito Zhang, Jinkai Hu, Jiayao Li, Rui Gao, Zekun Li, Songquan Zhu, Jingkai Zhou, and Pengyu Zhao. Minimax sparse attention, 2026. URL https://arxiv.org/abs/2606.13392.

Jingping Liu, Xingchen Peng, Yan Zhou, Ziyan Liu, Jie Zhai, Ronghao Chen, Huacan Wang, and Xiaofeng Jia. MirrorQA: Benchmarking multimodal LLMs on mirror-orientation reasoning. In Maria Liakata, Viviane P. Moreira, Jiajun Zhang, and David Jurgens (eds.), Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 40464–40476, San Diego, California, United States, July 2026. Association for Computational Linguistics. ISBN 979-8-89176-390-6. doi: 10.18653/v1/2026.acl-long.1879. URL https://aclanthology.org/2026.acl-long.1879/.

Lidong Lu, Guo Chen, Zhu Wei, Zhiqi Li, Yicheng Liu, and Tong Lu. Av-reasoner: Improving and benchmarking clue-grounded audio-visual counting for mllms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 33477–33487, June 2026.

Shiyin Lu, Yang Li, Qing-Guo Chen, Zhao Xu, Weihua Luo, Kaifu Zhang, and Han-Jia Ye. Ovis: Structural embedding alignment for multimodal large language model. arXiv:2405.20797, 2024.

George Mandler and Billie J. Shebo. Subitizing: An analysis of its component processes. Journal of Experimental Psychology: General, 111(1):1–22, 1982. ISSN 0096-3445. doi: 10.1037/ 0096-3445.111.1.1. URL http://dx.doi.org/10.1037/0096-3445.111.1.1.

Meta Superintelligence Labs. Muse-Glimmer-30B model card, 2026. URL https://huggingface.co/ meta-models/Muse-Glimmer-30B.

Mistral AI. Mistral small 4, 2026. URL https://mistral.ai/news/mistral-small-4/.

Anindya Mondal, Sauradip Nag, Xiatian Zhu, and Anjan Dutta. Omnicount: Multi-label object counting with semantic-geometric priors. Proceedings of the AAAI Conference on Artificial Intelligence, 39(18):19537–19545, April 2025. ISSN 2159-5399. doi: 10.1609/aaai.v39i18.34151. URL http://dx.doi.org/10.1609/aaai.v39i18.34151.

Gia Khanh Nguyen, Yifeng Huang, and Minh Hoai. Can current ai models count what we mean, not what they see? a benchmark and systematic evaluation, 2025. URL https://arxiv.org/abs/2509. 13939.

NVIDIA, :, Aditi, Niket Agarwal, Arslan Ali, Jon Allen, Martin Antolini, Adeline Aubame, Alisson Azzolini, Junjie Bai, Maciej Bala, Yogesh Balaji, Josh Bapst, Aarti Basant, Mukesh Beladiya, Mohammad Qazim Bhat, Zaid Pervaiz Bhat, Dan Blick, Vanni Brighella, Han Cai, Tiffany Cai, Eric Cameracci, Jiaxin Cao, Yulong Cao, Mark Carlson, Carlos Casanova, Ting-Yun Chang, Yan Chang, Yu-Wei Chao, Prithvijit Chattopadhyay, Roshan Chaudhari, Chieh-Yun Chen, Junyu Chen, Ke Chen, Qizhi Chen, Wenkai Chen, Xiaotong Chen, Yu Chen, An-Chieh Cheng, Click Cheng, Xiu Chia, Jeana Choi, Chaeyeon Chung, Wenyan Cong, Yin Cui, Magdalena Dadela, Nalin Dadhich, Wenliang Dai, Joyjit Daw, Alperen Degirmenci, Rodrigo Vieira Del Monte, Robert Denomme, Sameer Dharur, Marco Di Lucca, Ke Ding, Wenhao Ding, Yifan Ding, Yuzhu Dong, Nicole Drumheller, Yilun Du, Aigul Dzhumamuratova, Aleksandr Efitorov, Hamid Eghbalzadeh, Naomi Eigbe, Imad El Hanafi, Hassan Eslami, Benedikt Falk, Jiaojiao Fan, Jim Fan, Amol Fasale, Sergiy Fefilatyev, Liang Feng, Francesco Ferroni, Sanja Fidler, Xiao Fu, Vikram Fugro, Prashant Gaikwad, TJ Galda, Katelyn Gao, Yihuai Gao, Wenhang Ge, Sreyan Ghosh, Arushi Goel, Vivek Goel, Akash Gokul, Rama Govindaraju, Jinwei Gu, Miguel Guerrero, Elfie Guo, Aryaman Gupta, Siddharth Gururani, Hugo Hadfield, Song Han, Ankur Handa, Zekun Hao, Mohammad Harrim, Ali Hassani, Nathan Hayes-Roth, Yufan He, Chris Helvig, Cyrus Hogg, Madison Huang, Michael Huang, Sophia Huang, Yufan Huang, Jacob Huffman, DeLesley Hutchins, Suneel Indupuru, Boris Ivanovic, Arihant Jain, Joel Jang, Ryan Ji, Yanan Jian, Dongfu Jiang, Jingyi Jin, Atharva Joshi, Nikhilesh Joshi, Pranjali Joshi, Andy Ju, Jaehun Jung, Weiwei Kang, Scott Kassekert, Jan Kautz, Ashna Khetan, Julia Kiczka, Slawek Kierat, Gwanghyun Kim, Kuno Kim, Sunny Kim, Kezhi Kong, Xin Kong, Zhifeng Kong, Tomasz Kornuta, Egor Krivov, Hui Kuang, Saurav Kumar, Chia-Wen Kuo, George Kurian, Wojciech Kutak, JF Lafleche, Himangshu Lahkar, Omar Laymoun, Jayjun Lee, Sanggil Lee, Gabriele Leone, Boyi Li, Freya Li, Jiajun Li, Jinfeng Li, Ling Li, Pengcheng Li, Shangru Li, Tingle Li, Xiaolong Li, Xuan Li, Zhaoshuo Li, Zhiqi Li, Hao Liang, Maosheng Liao, Chen-Hsuan Lin, Tsung-Yi Lin, Ming-Yu Liu, Sifei Liu, Zihan Liu, Hai Loc Lu, Xiangyu Lu, Alice Luo, Ruipu Luo, Wenjie Luo, Jiangran Lyu, Martin Ding Ma, Nic Ma, Qianli Ma, Dawid Majchrowski, Louis Marcoux, Miguel Martin, Qing Miao, Ashkan Mirzaei, Shreyas Misra, Kaichun Mo, Durra Mohsin, Hyejin Moon, Pawel

Morkisz, Saeid Motiian, Kirill Motkov, Seungjun Nah, Yashraj Narang, Deepak Narayanan, Thabang Ngazimbi, Julian Ouyang, Shubham Pachori, David Page, Yatian Pang, Sehwi Park, Mahesh Patekar, Mostofa Patwary, Marco Pavone, Trung Pham, Wei Ping, Soha Pouya, Shrimai Prabhumoye, Varun Praveen, Delin Qu, Hesam Rabeti, Morteza Ramezanali, Marilyn Reeb, Xuanchi Ren, Kristen Rumley, Wojciech Rymer, Jun Saito, Yeongho Seol, John Shao, Piyush Shekdar, Tianwei Shen, Humphrey Shi, Min Shi, Stella Shi, Kevin Shih, Mohammad Shoeybi, Mateusz Sieniawski, Shuran Song, Alexander Sotelo, Amir Sotoodeh, Sunil Srinivasa, Vignesh Srinivasakumar, Bartosz Stefaniak, Rahul Heinrich Steiger, Shangkun Sun, Jiaxiang Tang, Shitao Tang, Yangyang Tang, Yue Tang, Tolou Tavakkoli, Kayley Ting, Krzysztof Tomala, Wei-Cheng Tseng, Jibin Varghese, Sergei Vasilev, Thomas Volk, Raju Wagwani, Roger Waleffe, Andrew Z. Wang, Boxiang Wang, Haoxiang Wang, Qiao Wang, Shihao Wang, Shijie Wang, Ting-Chun Wang, Yan Wang, Yu Wang, Rohit Watve, David Wehr, Fangyin Wei, Xinshuo Weng, Jay Zhangjie Wu, Kedi Wu, Hongchi Xia, Summer Xiao, Tianjun Xiao, Kevin Xie, Daguang Xu, Jiashu Xu, Mengyao Xu, Ruqing Xu, Xingqian Xu, Yao Xu, Dinghao Yang, Dong Yang, Hans Yang, Xiaodong Yang, Xuning Yang, Yichu Yang, Yurong You, Zhiding Yu, Hao Yuan, Simon Yuen, Xiaohui Zeng, Pengcuo Zeren, Cindy Zha, Haotian Zhang, Jenny Zhang, Jing Zhang, Liangkai Zhang, Paris Zhang, Shun Zhang, Xuanmeng Zhang, Zhizheng Zhang, Ann Zhao, Yilin Zhao, Yuliya Zhautouskaya, Charles Zhou, Fengzhe Zhou, Shilin Zhu, Yuke Zhu, Dima Zhylko, and Artur Zolkowski. Cosmos 3: Omnimodal world models for physical ai, 2026. URL https://arxiv.org/abs/2606.02800.

OpenAI. GPT-5.6 system card, July 2026a. URL https://deploymentsafety.openai.com/gpt-5-6.

OpenAI. GPT-6 Astra system card, September 2026b. URL https://deploymentsafety.openai.com/ gpt-6-astra.

Roni Paiss, Ariel Ephrat, Omer Tov, Shiran Zada, Inbar Mosseri, Michal Irani, and Tali Dekel. Teaching clip to count to ten. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3170–3180, October 2023.

Atin Pothiraj, Elias Stengel-Eskin, Jaemin Cho, and Mohit Bansal. Capture: Evaluating spatial reasoning in vision language models via occluded object counting. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 8001–8010, October 2025.

Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026a. URL https://qwen.ai/ blog?id=qwen3.5.

Qwen Team. Qwen3.6-35B-A3B: Agentic coding power, now open to all, April 2026b. URL https://qwen.ai/blog?id=qwen3.6-35b-a3b.

Qwen Team. Qwen3.8-Max: A new bar for coding and cowork, August 2026c. URL https://qwen. ai/blog?id=qwen3.8.

Viresh Ranjan, Udbhav Sharma, Thu Nguyen, and Minh Hoai. Learning to count everything. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 3394–3403, June 2021.

Chenggang Rong, Tao Han, Zhiyuan Zhao, Yaowu Fan, Jia Wan, Song Guo, Yuan Yuan, and Junyu Gao. Unicbench: Unified counting benchmark for mllm. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 23730–23740, June 2026.

Elizabeth S. Spelke. Principles of object perception. Cognitive Science, 14(1):29–56, January 1990. ISSN 1551-6709. doi: 10.1207/s15516709cog1401 3. URL http://dx.doi.org/10.1207/ s15516709cog1401 3.

Jayant Sravan Tamarapalli, Rynaa Grover, Nilay Pande, and Sahiti Yerramilli. Countqa: How well do mllms count in the wild?, 2025. URL https://arxiv.org/abs/2508.06585.

Gemma Team. Gemma 4 technical report, 2026a. URL https://arxiv.org/abs/2607.02770.

IBM Granite Vision Team. Granite 4.1 vision, 2026b. URL https://huggingface.co/ibm-granite/ granite-vision-4.1-4b.

Fumihiko Tsuchiya, Taiki Miyanishi, Shunsuke Yasuki, Mahiro Ukai, Nakamasa Inoue, Shuhei Kurita, Yusuke Iwasawa, and Yutaka Matsuo. Diagnosing long-video quantitative reasoning in multimodal llms via enumeration and counting, 2026. URL https://arxiv.org/abs/2603.29943.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, Jingjing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. Internvl3.5: Advancing open-source multimodal models in versatil ity, reasoning, and efficiency, 2025. URL https://arxiv.org/abs/2508.18265.

Tengjin Weng, Jingyi Wang, Wenhao Jiang, and Zhong Ming. Visnumbench: Evaluating number sense of multimodal large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 3830–3840, October 2025.

Karen Wynn. Addition and subtraction by human infants. Nature, 358(6389):749–750, August 1992. ISSN 1476-4687. doi: 10.1038/358749a0. URL http://dx.doi.org/10.1038/358749a0.

Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.

## A SCENE CURATION AND ANNOTATION VERIFICATION

Section 3.2 describes the construction of PhysCount. This section focuses on two boundaries that require further detail: when a mirror or glass scene genuinely changes the count, and how physical identity is established when no direct appearance is visible. We then describe the QA audit and provide complete annotation examples.

## A.1 CANDIDATE SCENE BOUNDARIES

Candidate images are drawn from Common Canvas Gokaslan et al. (2024), GPIC Chandrasegaran et al. (2026), and other public collections. We retain two scene types: reflection produces a clear duplicate of a countable target, or targets behind glass overlap with salient surface reflections in a way that changes target boundaries or apparent multiplicity. A reflective surface alone is insufficient, and the target must remain reliably countable after careful inspection.

The curation criteria separate physical existence from the path of observation. Objects seen directly or through a transparent medium are treated as real objects; only appearances formed on a reflective surface are reflections. Content displayed on televisions, posters, photographs, or paintings is excluded, as are shadows, highlights, and weak reflections on wet surfaces. One entity together with a complete mirror appearance is sufficient to create meaningful appearance duplication. Ordinary transmission is not: an object merely seen through glass does not enter the benchmark unless a surface reflection also introduces duplication, overlap, or an identity relation.

Qwen3.6-35B-A3B Qwen Team (2026b) evaluates each candidate twice. The first pass identifies the counting target and checks countability, reflection salience, and counting ambiguity. Only candidates satisfying every condition proceed to a second judgment. The second pass receives the image and target category without the first-pass rationale, and independently verifies the duplicated or mixed reflection–transmission structure. Both records are validated as structured JSON, and a fixed conjunction of their fields determines retention.

## A.1.1 FIRST-PASS SCREENING

The first pass rejects weak or incomplete reflections, reflective content unrelated to the target, and scenes that do not support reliable human counting. Source captions serve only as retrieval cues; visible evidence determines the decision.

## PROMPT Stage-one screening

You are a strict data screener for a visual counting benchmark. Make judgments   
only from evidence visible in the current image. Do not rely on information in   
the caption that is not visible in the image.   
Your task is to find real images in which reflection or transmission creates a   
clear counting structure. Prefer rejecting many candidates over retaining a   
weakly relevant image.   
Here, "counting ambiguity" does not require a reflection to completely deceive   
the viewer. It means that reflection/transmission creates salient repetition,   
superposition, partitioning, or occlusion, such that REAL and   
REFLECTION/TRANSMISSION must be distinguished to count correctly.   
- REAL: a real object seen directly or through a transparent medium.   
- REFLECTION: an appearance reflected by a mirror, glass, water, a glossy floor,   
or a metallic surface.   
- An object seen through glass remains REAL; only the reflected appearance   
superimposed on the glass surface is REFLECTION.   
- Images shown on televisions, posters, photographs, paintings, or screens are   
not reflections. Shadows are not reflections either.   
Set KEEP=YES only if at least one of the following paths is satisfied:   
A. Duplicated appearance: a reflection contains a clearly delineated repeated   
appearance of the same category that an ordinary viewer might count twice.   
B. Reflection-transmission mixture: real countable targets behind a shop   
window, glass door, display case, or similar surface significantly overlap,   
join, shift relative to, or occlude reflections on the glass surface, truly

creating missed counts, double counts, or ambiguous target boundaries.

## Both paths must satisfy all of the following:

1. The reflection is sufficiently salient, rather than a weak highlight, a locally blurred reflection, or an incidental background detail.

2. A genuine counting ambiguity exists.

3. After careful inspection, a human can still reliably identify the primary counting target.

## The following structures are positive examples calibrated from human-selected cases:

\- Mirror walls, mirrored columns, or multiple mirrors copy the same person or object into multiple regions.

\- Real people, products, mannequins, or other objects inside a shop window are clearly superimposed with reflected street scenes or people on the glass.

\- Glass in a shop, cafe, or display case simultaneously presents countable interior targets and strong exterior reflections.

\- A person stands before a mirror, and both the real person and the reflected person are complete, salient, and separately countable.

\- Even when there is only one real person, retain the image if the person and a complete, clear mirror appearance are visible simultaneously.

Multi-faceted mirrors, mirror installations, or highly reflective objects produce multiple clear copies.

\- An object placed on a mirrored surface and its complete reflection form two unambiguous appearances.

## High-value cases include:

1. Targets of the same category occur as both REAL and REFLECTION.

2. One real target produces multiple reflected appearances.

3. Real targets behind glass and reflections on the glass surface coexist.

4. Reflection/transmission causes overlap, displacement, occlusion, or joining that can induce double counting or missed counting.

5. The image supports separate questions about the number of real objects, reflected appearances, real objects seen through a transparent medium, and total visual appearances.

6. There are many targets, but humans can still count them reliably.

## Always set KEEP=NO for the following:

\- Targets are merely seen through water or glass, without a clear reflected duplicate of the same category.

\- Reflections on wetlands, stage floors, or water are blurred, incomplete, or very faint.

\- A reflection only indicates that a surface is reflective and would not be mistaken for an additional target.

\- The real and reflected targets belong to different categories and there is no salient reflection-transmission mixture.

The reflection is distant, tiny, or inconspicuous and does not affect an initial count.

\- There is only a faint reflection on water, wet ground, or an ordinary floor, without a clear duplicated structure.

\- Fish are merely seen through water or people are merely seen through glass, with no strong reflection superimposed on the surface.

## The following must also be rejected:

\- Only lights, sky, architectural textures, or abstract colors are reflected, with no clear countable target.

\- A mirror or glass surface is present but irrelevant to counting the target.

\- Targets are too small, blurred, or heavily occluded for reliable human counting.

\- Product photographs consisting only of a mirror, glass, water, or metal.

\- Illustrations, diagrams, logos, text pages, purely abstract images, collages, or multi-panel images.

\- No single clear primary counting category can be identified.

## Judgment requirements:

Select only one most suitable primary counting category; do not mix multiple categories.

\- countability measures whether a human can reliably identify targets one by

\- reason must be concise and no longer than one sentence.

Output only the complete JSON required by the schema. Do not provide an

The response records the proposed target and scene type together with judgments of countability, reflection salience, and counting ambiguity. Every condition must be satisfied before verification.

## A.1.2 INDEPENDENT VERIFICATION

The verification pass reassesses whether the scene is photographic, the target is countable, and reflection creates either genuine duplication or a consequential interaction with transmission. Rejected candidates are assigned to fixed categories covering non-photographic content, mediated images, abstract reflections, irrelevant glass, unclear or uncountable targets, and structurally simple scenes.

![](images/7ebf1492c41946675c9427b5d2111aef3f9c684bbf476857909d02d434ca9803.jpg)

## A.1.3 FINAL SELECTION

The final decision uses no additional model score. A retained image must satisfy every first-pass condition and receive an affirmative verification decision with high countability, counting value, and reflection salience, confirmed counting ambiguity, and no false-positive type.

Ordinary transmission does not pass this rule even when the transmitted targets are clear. Through-glass appearances remain eligible inside a mixed reflection–transmission scene. This distinction ensures that every retained image contains an image-formation effect that changes the correspondence between visible appearances and physical entities.

## A.2 ANNOTATION VERIFICATION

A physical entity need not have a direct appearance in the image. Several mirror-only appearances may belong to one entity, while direct, reflected, and through-glass appearances may all refer to the same object. Multi-mirror identities are recorded as complete same-instance groups rather than disconnected pairwise links. When an image contains several suitable target categories, each category forms a separate annotation group.

The first annotator marks appearance locations and modes, groups appearances by physical identity, and writes the questions and answers from the image. A second annotator checks the appearance locations, image-formation labels, and identity groups against the source image. GPT-5.6 Sol then recomputes each answer from the verified annotation and checks whether the question meaning agrees with that calculation. A third annotator resolves every flagged case.

Appearance questions count annotated observations; entity questions count physical objects or identity groups; image-formation questions select observations by mode; and correspondence questions compute intersections, exclusions, and multiplicities over identity groups. This shared annotation makes every answer auditable from the same scene record.

## A.2.1 QUESTION–ANSWER CONSISTENCY AUDIT

For each question, the auditor identifies the requested counting unit, selects the relevant appearances or identity groups, and derives the expected answer through an explicit set operation. The audit covers total and mode-specific appearances, distinct entities, entities defined by their available appearance types, reflection multiplicity, and counterfactual changes to an image-formation path.

The derived answer is compared directly with the recorded QA. Agreement is marked correct; a supported question with a conflicting answer is marked contradicted. Ambiguous questions, image–annotation conflicts, and insufficient evidence are sent to manual review, while questions requiring an unannotated attribute are marked unsupported. A semantically correct but unnatural question may be rewritten without changing its answer and is marked revised.

## PROMPT Question–answer consistency audit

You are auditing one visual counting annotation group for a benchmark about   
physical objects, mirror reflections, and transparent-media views.   
You receive:   
1. The source image.   
2. An optional overlay image showing annotated points, labels, identity groups,   
and relations.   
3. The structured annotation for one group.   
4. Existing question-answer pairs written for this unit.   
Your task is to independently verify every QA using the image and annotation.   
Do not invent objects, relations, image-formation paths, or quantities that   
are not supported by the supplied evidence.   
IMPORTANT CONCEPTS   
- An appearance is one annotated visual occurrence in the image.   
A physical-space appearance, a mirror reflection appearance, and a   
transmission appearance are all appearances.   
- A physical entity is a distinct real-world object after merging all   
appearances joined by the recorded identity links.   
- A physical-space appearance is NOT automatically the same as a distinct   
physical entity. A real object may be visible only in a mirror or through   
glass; conversely, one entity may create multiple reflection appearances.   
- The recorded identity grouping has priority when deciding whether   
appearances belong to the same object.   
- Labels indicate image-formation mode:   
- direct: an appearance occupying physical space

reflection: an appearance formed by a mirror or reflective surface   
through-glass: an appearance seen through a transparent medium   
For questions about "different", "distinct", "actual objects in the scene",   
or wording that explicitly merges an object and its reflections, count   
physical entities, not appearances.   
For questions about what "occupies physical space", count only   
physical-space appearances. Do not silently deduplicate them into entities.   
For questions about objects visible only in a mirror, count identity groups   
that have reflection appearance(s) but no physical-space appearance.   
For objects visible only physically, count identity groups that have   
physical-space appearance(s) but no reflection appearance.   
For objects visible both physically and in a mirror, count identity groups   
that have both.   
For questions about the maximum number of mirror images of one object,   
compute the maximum number of reflection appearances linked to any one   
identity group.   
AUDIT PROCEDURE FOR EACH QA   
1. Read the question literally and classify its count target.   
2. Identify the exact relevant appearances or entity groups in the annotation.   
3. Compute the expected numerical answer from the annotation.   
4. Check the source image and overlay for obvious contradictions, ambiguity,   
annotation mistakes, or insufficient evidence.   
5. Compare the expected answer with the existing answer.   
6. Improve the question wording only if needed. Keep it natural, precise, and   
a counting question. Do not reveal the solution method or leak hidden   
annotation facts into the wording.   
7. Return a verdict and an explicit derivation.   
VERDICT RULES   
correct:   
The question is natural enough, the annotation supports the required count, and   
the existing answer matches the derived answer.   
- revised:   
The question wording requires revision, but its meaning is supported and the   
existing numerical answer matches the derived answer.   
contradicted:   
The question is answerable from the annotation, but its existing answer conflicts   
with the derived answer.   
needs\_manual\_review:   
The count cannot be confirmed safely because the image, overlay, question   
semantics, or annotation creates genuine ambiguity or conflict.   
unsupported:   
The question requires information not represented by the annotation, so the   
answer cannot be calculated reliably.   
OUTPUT REQUIREMENTS   
For each question, return the verdict, the verified answer, any revised   
wording, the counting target, and an explicit derivation. For example:   
"Entities with direct appearances: {E1, E2, E3}.   
Entities with both direct and reflected appearances: {E1, E3}.   
Therefore, only E2 is visible directly but not by reflection, so the answer is 1."   
Name the relevant appearances, entity groups, relations, or set operation;   
do not use vague claims such as "the annotation indicates this".

![](images/72de2baa829b27720a6fef504c91793e38d38b9d8c92eab470059f3e11cfc6e1.jpg)

## A.3 WORKED ANNOTATION AND QA EXAMPLES

Figures 7–12 show all questions from six reviewed annotation groups. Each figure pairs the source image with its appearance–entity overlay. Blue D nodes mark direct physical-space appearances, orange R nodes mark reflected appearances, and green T nodes mark appearances seen through transparent media. Shared entity identifiers and connecting lines show which appearances depict the same object. The compact diagram below each image summarizes these identities, and the table gives every QA and its verified answer. Category A counts all visual appearances, E counts objects occupying physical space or distinct physical entities, I counts appearances by image-formation mode, and C queries relations between appearances and identity groups.

<table><tr><td>Category</td><td>Question</td><td>Answer</td></tr><tr><td>A</td><td>Counting both the cats physically present in the photo, including any seen through the glass door, and the cat reflec- tions on the glass door, how many times do cats appear in total?</td><td>3</td></tr><tr><td>E</td><td>How many different cats are actually in the scene photographed?</td><td>2</td></tr><tr><td>I</td><td>How many cat reflections are visible on the glass door?</td><td>1</td></tr><tr><td>I</td><td>How many cats can be seen through the glass door that extends horizontally across the image?</td><td>1</td></tr><tr><td>C</td><td>Among the cats on the same side of the glass door as the photographer, how many do not form a reflection on the glass door?</td><td>0</td></tr><tr><td>C</td><td>What is the maximum number of reflections of a single cat on the glass door?</td><td>1</td></tr></table>

Figure 7: Example 1: reflection and transmission of the same target category. Three cat appearances form two physical identities: one identity is observed directly and by reflection, while the other is seen through glass. The table reports every question derived from these annotations.

![](images/e65881a35c2a436d71dcef63db74b96433265ffa6f7d5ee330c4e0d878ce577e.jpg)

<table><tr><td>Category</td><td>Question</td><td>Answer</td></tr><tr><td>A</td><td>How many visible appearances of people are there in total in the photo, including the people on the stage and their reflections in the mirror?</td><td>3</td></tr><tr><td>E</td><td>How many people physically occupy space on the stage, excluding their reflections in the mirror?</td><td>1</td></tr><tr><td>E</td><td>How many different people are actually in the scene captured in this photo?</td><td>1</td></tr><tr><td>I</td><td>How many people&#x27;s reflections are shown in the mirror in total?</td><td>2 0</td></tr><tr><td>CC</td><td>How many people appear only in the physical space of the stage, with no reflection in the mirror? How many people are visible only as reflections in the mirror, with no corresponding person visible in the physical</td><td>0</td></tr><tr><td></td><td>space of the stage?</td><td></td></tr><tr><td>CC</td><td>How many people appear both in the physical space of the stage and in the mirror?</td><td>1</td></tr><tr><td></td><td>What is the maximum number of reflections of the same person shown in the mirror?</td><td>2</td></tr></table>

Figure 8: Example 2: one person with two reflected appearances. The direct appearance and both mirror appearances share one identity. The questions contrast appearance enumeration with physical occupancy, identity, and reflection multiplicity.

![](images/5bc4cda934479ae825c5089abfbbbabcfa27b5ccd7d6721c8259459d60efdcd4.jpg)  
Figure 9: Example 3: one person distributed across three mirror regions. Four appearances map to one physical identity, enabling questions about total appearances, reflected subsets, physical– reflection overlap, and maximum multiplicity.

![](images/d1c4828e33531365c97015b14c7e46f0d75a10ef6c4682d063934bda20d98b95.jpg)

Appearance-to-entity annotatior  
![](images/9dbc663e9f741bb3d6a06ed680b6a5f70db29361861b9fe27b58d83069d5a5ca.jpg)

<table><tr><td>Category</td><td>Question</td><td>Answer</td></tr><tr><td>A</td><td>Counting both the dice physically present in the photo and the mirror reflections, how many dice appear in total?</td><td>7</td></tr><tr><td>E</td><td>How many dice are physically present in the photo?</td><td>1</td></tr><tr><td>E</td><td>How many distinct dice are actually in the scene captured by this photo?</td><td>1</td></tr><tr><td>I</td><td>How many mirror reflections of dice are there in the photo?</td><td>6</td></tr><tr><td>C</td><td>How many dice have more than one reflection in the mirror?</td><td>1</td></tr><tr><td>CC</td><td>How many dice are physically present in the photo but do not appear in the mirror?</td><td>0</td></tr><tr><td></td><td>What is the maximum number of reflections formed in the mirror by the same die?</td><td>6</td></tr></table>

Figure 10: Example 4: high reflection multiplicity from a single object. One physical die produces six reflected appearances. The resulting questions separate the seven visible appearances from one physical identity and its reflection multiplicity.

Appearance-to-entity annotation  
![](images/705c2fc3b3a1847f0f3386b5b194e7bfc249a9e2bf13cc04f37aa40066231a72.jpg)

<table><tr><td>Category</td><td>Question</td><td>Answer</td></tr><tr><td>A</td><td>How many shoes appear in total, counting both the shoes occupying physical space in the photo and their reflections in the mirror on the right?</td><td>4</td></tr><tr><td>E</td><td>How many shoes in the photo occupy physical space?</td><td>22</td></tr><tr><td>E</td><td>How many different shoes are actually in the scene captured in this photo?</td><td></td></tr><tr><td>I C</td><td>How many shoes are reflected in the mirror on the right?</td><td>2 0</td></tr><tr><td>C</td><td>How many shoes occupy physical space but have no corresponding reflection in the mirror on the right? How many shoes appear only in the mirror on the right, with no corresponding shoes occupying physical space in the</td><td>0</td></tr><tr><td></td><td>photo?</td><td></td></tr><tr><td>C</td><td>How many shoes both occupy physical space in the photo and appear as reflections in the mirror on the right?</td><td>2</td></tr><tr><td>C</td><td>What is the maximum number of times the same shoe appears as a reflection in the mirror on the right?</td><td>1</td></tr></table>

Figure 11: Example 5: two physical objects paired with their mirror appearances. Four shoe appearances form two identity groups. The questions require two independent direct–reflection correspondences rather than a single global count.

D×1 R×2 ⇒ $E _ { 1 } = \{ D _ { 1 } , R _ { 1 } , R _ { 2 } \}$  
Original  
Appearance-to-entity annotatior  
![](images/49c080d9c18a826f484e4f4760d3a6d6443edb5fbcc43f3e3b452e2cfcf55b81.jpg)

<table><tr><td>Category</td><td>Question</td><td>Answer</td></tr><tr><td>A</td><td>How many cameras appear in total in the photo?</td><td>3</td></tr><tr><td>E</td><td>How many cameras occupy physical positions in the photo?</td><td>1</td></tr><tr><td>E</td><td>How many different cameras are in the photo?</td><td>1</td></tr><tr><td>I</td><td>How many cameras appear in the mirror in the photo?</td><td>2</td></tr><tr><td>C</td><td>How many cameras appear both in a physical position and in the mirror?</td><td>1</td></tr><tr><td>C</td><td>What is the maximum number of mirror images that any single camera has in the photo at the same time?</td><td>2</td></tr></table>

Figure 12: Example 6: a direct camera repeated across two mirrors. Three camera appearances belong to one physical identity. The questions distinguish the direct camera, the reflected subset, their shared identity, and reflection multiplicity.

## B EVALUATION DETAILS AND REPRODUCIBILITY

## B.1 PROMPTING PROTOCOLS

Independent QA. Each image–question pair is evaluated in a separate model call. The image and the following prompt are supplied together in one user message:

PROMPT Independent question answering   
Answer the following counting question about this image. Carefully distinguish   
visual appearances from physical entities, including appearances formed by   
reflections or seen through glass, and use appearance-entity correspondences   
when required by the question. Follow the counting target specified in the   
question. Return exactly one non-negative integer, with no labels or prose.   
Question: {question}

Questions from the same annotation group are answered without access to one another. No additional system instruction is used.

Group-level joint prompting. All factual questions from one annotation group are placed in a single prompt with one shared image, and the answers follow the question order.

PROMPT Group-level joint questioning   
Answer the following counting questions about this image. Carefully distinguish   
visual appearances from physical entities, including appearances formed by   
reflections or seen through glass, and use appearance-entity correspondences   
when required by each question. Follow the counting target specified in each   
question. Return exactly one JSON array of non-negative integers in the same   
order as the {N} questions, with no labels or prose.   
1. {question\_1}   
2. {question\_2}   
N. {question\_N}

The two conditions use the same task instructions, model configuration, and image preprocessing. They differ in whether questions from an annotation group are supplied separately or together and, correspondingly, whether the response is one integer or an ordered integer array.

## B.2 IMAGE AND INFERENCE SETTINGS

Images retain their native resolution up to the specified long-edge limit $L ;$ larger images are resized isotropically without cropping or upsampling. The resolution study uses $L ^ { \bar { ~ } } \in$ {512, 1024, 2048, 4096}, and all other experiments use $L = 4 0 9 6$ . For each model, we select the strongest reasoning or thinking option exposed by the evaluated backend: this is high when that is the highest available level, xhigh when supported, and the model’s native thinking mode when no graded effort control is provided. These backend-specific labels describe the selected operating mode and are not assumed to be numerically equivalent. We cap generation at 32,768 tokens where supported. Temperature 1.0, top p=0.95, and top k=20 are applied only when the backend exposes the corresponding controls; unsupported fields are omitted.

The human baseline follows the independent-QA setting. Five participants each answer the complete set of 4,595 questions, receiving the same image and question shown to a model on every trial. We compute all metrics separately for each participant and report their arithmetic mean.

## B.3 ANSWER VALIDATION

Independent evaluation accepts one non-negative integer; joint evaluation accepts an integer array aligned with the question order. Missing, malformed, negative, or extreme predictions are treated as invalid: they receive maximal NAE and fail exact-match metrics, but are excluded from raw-count MAE.

## C EXPANDED DIAGNOSTIC RESULTS

This section provides the numerical results behind the diagnostic analyses in Section 4.3. All values follow the same group-macro evaluation protocol as the main results.

## C.1 PERFORMANCE ACROSS COUNTING TARGETS

The target-domain analysis in Figure 4 shows that object recognition and physical-entity reasoning remain coupled. A model that handles people reliably may still fail on compact manufactured objects, furniture, or irregular natural targets, and these shifts are not simple rescalings of overall performance. Qwen3.8-27B is particularly strong on several compact categories, Gemini 3.7 Flash is more stable across human-like and animal targets, and Gemma 4-31B peaks on electronics. Table 4 reports the corresponding Scores for all 40 models.

## C.2 APPEARANCE MULTIPLICITY

Appearance multiplicity is the largest number of annotated appearances assigned to one physical entity in a group. Performance changes little across the first three bins, then breaks when an entity generates at least four appearances: mean Entity NAE rises, mean Score falls to its minimum, and 32 of 40 models score below their single-appearance result. The full trajectories in Table 5 show that the difficulty comes from maintaining a compatible identity assignment across many observations, rather than from a uniform increase in raw count.

## C.3 WITHIN- AND CROSS-MODE CORRESPONDENCE

The within-/cross-mode comparison retains groups with repeated appearances and standardizes both conditions to the same multiplicity distribution. Within-mode identities repeat inside one observation mode; cross-mode identities span at least two of direct, reflected, and through-glass observation. Table 6 shows higher cross-mode correspondence error for 31 of 40 models. Score follows a different pattern because direct views support detection and enumeration even when matching them to indirect appearances remains difficult. The comparison therefore separates access to a visual anchor from the ability to preserve identity across image-formation paths.

Table 4: Performance across counting-target domains. Each entry is the group-macro Score (%) for one counting-target domain.
<table><tr><td>Model</td><td>People</td><td>Containers</td><td>Lighting</td><td>Art</td><td>Furniture</td><td>Architecture</td><td>Personal</td><td>Household</td><td>Electronics</td><td>Vehicles</td><td>Sports</td><td>Toys</td><td>Plants</td><td>Food</td><td>Animals</td><td>Other</td></tr><tr><td>Gemini 3.7 Flash</td><td>39.43</td><td>27.53</td><td>30.85</td><td>41.06</td><td>20.09</td><td>21.91</td><td>17.33</td><td>15.54</td><td>45.31</td><td>25.51</td><td>13.78</td><td>38.13</td><td>7.51</td><td>47.51</td><td>47.56</td><td>48.89</td></tr><tr><td>Muse-Glimmer-30B</td><td>30.16</td><td>23.31</td><td>26.72</td><td>34.90</td><td>27.08</td><td>26.96</td><td>17.12</td><td>20.59</td><td>56.25</td><td>22.04</td><td>23.84</td><td>39.23</td><td>20.65</td><td>41.39</td><td>49.34</td><td>29.19</td></tr><tr><td>Qwen3.8-27B</td><td>29.47</td><td>22.60</td><td>21.48</td><td>42.57</td><td>12.88</td><td>16.19</td><td>23.97</td><td>10.10</td><td>56.25</td><td>26.78</td><td>18.68</td><td>47.78</td><td>21.11</td><td>71.31</td><td>39.61</td><td>38.68</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>28.57</td><td>20.44</td><td>22.68</td><td>40.11</td><td>25.75</td><td>29.82</td><td>29.77</td><td>10.56</td><td>27.08</td><td>19.15</td><td>15.68</td><td>34.57</td><td>24.26</td><td>48.07</td><td>48.26</td><td>25.35</td></tr><tr><td>Gemma 4-31B</td><td>27.88</td><td>19.67</td><td>16.46</td><td>35.00</td><td>21.57</td><td>11.89</td><td>8.63</td><td>2.62</td><td>75.00</td><td>19.98</td><td>16.85</td><td>29.48</td><td>10.90</td><td>42.15</td><td>31.87</td><td>45.24</td></tr><tr><td>Qwen3.5-27B</td><td>27.11</td><td>19.04</td><td>21.06</td><td>42.99</td><td>29.98</td><td>18.25</td><td>16.47</td><td>14.72</td><td>20.31</td><td>25.10</td><td>20.87</td><td>34.59</td><td>10.23</td><td>28.75</td><td>27.54</td><td>35.51</td></tr><tr><td>Qwen3.5-122B-A10B</td><td>27.24</td><td>20.90</td><td>21.40</td><td>40.81</td><td>22.07</td><td>13.06</td><td>7.66</td><td>9.85</td><td>25.00</td><td>19.90</td><td>11.26</td><td>41.45</td><td>11.82</td><td>45.11</td><td>29.61</td><td>34.26</td></tr><tr><td>Qwen3.5-35B-A3B</td><td>27.24</td><td>17.79</td><td>19.91</td><td>34.48</td><td>19.30</td><td>11.93</td><td>19.94</td><td>14.04</td><td>31.25</td><td>18.12</td><td>8.45</td><td>34.01</td><td>9.49</td><td>35.81</td><td>27.42</td><td>51.02</td></tr><tr><td>Gemma 4-26B-A4B</td><td>24.87</td><td>19.67</td><td>6.42</td><td>34.09</td><td>18.81</td><td>14.49</td><td>8.49</td><td>14.25</td><td>37.50</td><td>23.95</td><td>9.98</td><td>38.24</td><td>5.87</td><td>39.48</td><td>26.38</td><td>55.99</td></tr><tr><td>Qwen3.5-9B</td><td>24.45</td><td>15.55</td><td>21.56</td><td>34.52</td><td>21.41</td><td>13.32</td><td>19.61</td><td>9.10</td><td>43.75</td><td>18.99</td><td>8.17</td><td>30.02</td><td>4.36</td><td>43.14</td><td>32.64</td><td>25.55</td></tr><tr><td>GPT-5.6 Sol</td><td>21.78</td><td>18.73</td><td>24.34</td><td>34.84</td><td>15.13</td><td>16.04</td><td>11.48</td><td>15.90</td><td>25.00</td><td>14.72</td><td>6.47</td><td>32.57</td><td>12.68</td><td>32.98</td><td>37.07</td><td>48.33</td></tr><tr><td>InternVL3.5-241B-A28B</td><td>21.24</td><td>14.76</td><td>23.80</td><td>24.58</td><td>11.64</td><td>9.56</td><td>17.00</td><td>10.53</td><td>12.50</td><td>24.02</td><td>25.73</td><td>32.95</td><td>11.39</td><td>39.28</td><td>46.71</td><td>17.82</td></tr><tr><td>DeepSeek-V4-Flash-Vision</td><td>20.90</td><td>16.49</td><td>22.48</td><td>25.08</td><td>20.96</td><td>16.35</td><td>4.97</td><td>20.50</td><td>37.50</td><td>19.37</td><td>6.00</td><td>20.61</td><td>9.26</td><td>36.37</td><td>36.58</td><td>33.75</td></tr><tr><td>GPT-5.6 Terra</td><td>20.85</td><td>17.57</td><td>19.08</td><td>19.90</td><td>16.87</td><td>19.00</td><td>16.99</td><td>6.46</td><td>51.04</td><td>20.00</td><td>11.48</td><td>14.93</td><td>12.88</td><td>22.78</td><td>30.43</td><td>41.94</td></tr><tr><td>Gemma 4-12B</td><td>19.28</td><td>16.44</td><td>16.00</td><td>23.37</td><td>22.75</td><td>10.88</td><td>9.01</td><td>12.90</td><td>15.62</td><td>20.57</td><td>2.38</td><td>33.56</td><td>5.15</td><td>40.49</td><td>22.53</td><td>28.16</td></tr><tr><td>Qwen3.5-4B</td><td>19.06</td><td>15.27</td><td>17.91</td><td>28.31</td><td>18.75</td><td>8.90</td><td>16.15</td><td>3.09</td><td>4.69</td><td>22.17</td><td>7.84</td><td>29.65</td><td>3.14</td><td>25.83</td><td>23.48</td><td>17.21</td></tr><tr><td>InternVL3.5-30B-A3B</td><td>18.25</td><td>11.61</td><td>20.44</td><td>23.81</td><td>10.11</td><td>8.18</td><td>16.92</td><td>6.92</td><td>12.50</td><td>26.12</td><td>19.84</td><td>17.86</td><td>13.73</td><td>26.73</td><td>21.84</td><td>13.98</td></tr><tr><td>MiniMax-M3-428B-A23B</td><td>17.93</td><td>11.22</td><td>12.37</td><td>22.43</td><td>12.29</td><td>8.13</td><td>14.16</td><td>23.52</td><td>12.50</td><td>12.94</td><td>21.33</td><td>17.80</td><td>6.00</td><td>19.82</td><td>31.21</td><td>44.79</td></tr><tr><td>GPT-5.6 Luna</td><td>17.28</td><td>11.53</td><td>16.29</td><td>20.24</td><td>10.19</td><td>17.37</td><td>12.90</td><td>13.49</td><td>0.00</td><td>18.38</td><td>2.25</td><td>24.30</td><td>11.68</td><td>40.34</td><td>26.89</td><td>24.44</td></tr><tr><td>InternVL3.5-38B</td><td>15.96</td><td>14.69</td><td>6.78</td><td>22.51</td><td>16.88</td><td>8.84</td><td>20.69</td><td>4.57</td><td>12.50</td><td>12.73</td><td>13.73</td><td>19.74</td><td>3.06</td><td>24.17</td><td>18.57</td><td>22.47</td></tr><tr><td>InternVL3.5-14B</td><td>15.49</td><td>11.14</td><td>12.24</td><td>23.80</td><td>9.62</td><td>10.94</td><td>6.91</td><td>9.91</td><td>6.25</td><td>19.78</td><td>23.41</td><td>17.40</td><td>12.81</td><td>13.40</td><td>20.59</td><td>27.78</td></tr><tr><td>Molmo2-4B</td><td>15.35</td><td>11.75</td><td>6.49</td><td>18.60</td><td>13.85</td><td>12.90</td><td>5.11</td><td>3.47</td><td>31.25</td><td>20.50</td><td>9.25</td><td>19.98</td><td>4.74</td><td>11.07</td><td>31.83</td><td>24.57</td></tr><tr><td>Ovis2.6-80B-A3B</td><td>15.39</td><td>13.36</td><td>12.34</td><td>11.66</td><td>17.12</td><td>9.27</td><td>6.68</td><td>6.90</td><td>6.25</td><td>15.59</td><td>4.05</td><td>21.30</td><td>5.95</td><td>15.44</td><td>36.06</td><td>12.07</td></tr><tr><td>MiniCPM-V 4.5</td><td>14.98</td><td>10.70</td><td>8.21</td><td>22.25</td><td>11.03</td><td>10.56</td><td>16.31</td><td>14.61</td><td>43.75</td><td>11.31</td><td>20.43</td><td>13.33</td><td>10.56</td><td>18.62</td><td>27.27</td><td>22.63</td></tr><tr><td>Cosmos3-Super</td><td>15.01</td><td>9.19</td><td>16.08 10.37</td><td>30.66 16.07</td><td>9.87 19.48</td><td>8.14 10.67</td><td>13.25 16.81</td><td>10.84 1.35</td><td>18.75 6.25</td><td>16.79 15.41</td><td>5.48 6.38</td><td>12.27 13.51</td><td>7.40 22.37</td><td>19.89 10.42</td><td>15.79 11.28</td><td>19.45 28.05</td></tr><tr><td>InternVL3.5-8B Molmo2-8B</td><td>15.07 13.47</td><td>6.59 11.39</td></table>

Table 5: Performance across appearance multiplicity. Each entry is a group-macro percentage within the indicated maximum number of appearances per entity. The final bin produces the clearest rise in Entity NAE and the broadest decline in Score.
<table><tr><td rowspan="2">Model</td><td colspan="4">Entity NAE↓</td><td colspan="4">Score↑</td></tr><tr><td>1</td><td>2</td><td>3</td><td>≥4</td><td>1</td><td>2</td><td>3</td><td>≥4</td></tr><tr><td>Gemini 3.7 Flash</td><td>27.55</td><td>26.58</td><td>26.29</td><td>32.54</td><td>40.54</td><td>35.27</td><td>31.19</td><td>30.07</td></tr><tr><td>Muse-Glimmer-30B</td><td>35.32</td><td>31.36</td><td>27.56</td><td>26.38</td><td>28.72</td><td>29.68</td><td>29.51</td><td>33.27</td></tr><tr><td>Qwen3.8-27B</td><td>35.75</td><td>30.19</td><td>27.70</td><td>36.07</td><td>29.98</td><td>28.19</td><td>30.26</td><td>26.61</td></tr><tr><td>Qwen3.5-397B</td><td>33.20</td><td>32.50</td><td>25.23</td><td>45.92</td><td>27.71</td><td>27.89</td><td>31.36</td><td>26.48</td></tr><tr><td>Gemma 4-31B</td><td>33.47</td><td>37.28</td><td>34.66</td><td>51.63</td><td>29.73</td><td>24.68</td><td>25.27</td><td>22.34</td></tr><tr><td>Qwen3.5-27B</td><td>32.55</td><td>35.09</td><td>32.26</td><td>42.57</td><td>27.94</td><td>25.44</td><td>23.55</td><td>27.06</td></tr><tr><td>Qwen3.5-122B</td><td>32.87</td><td>34.29</td><td>31.68</td><td>44.46</td><td>27.75</td><td>24.82</td><td>26.23</td><td>25.93</td></tr><tr><td>Qwen3.5-35B</td><td>30.84</td><td>34.41</td><td>28.79</td><td>43.00</td><td>29.25</td><td>23.37</td><td>25.07</td><td>25.32</td></tr><tr><td>Gemma 4-26B-A4B</td><td>34.60</td><td>39.04</td><td>40.33</td><td>60.37</td><td>28.71</td><td>21.84</td><td>20.00</td><td>17.39</td></tr><tr><td>Qwen3.5-9B</td><td>35.80</td><td>34.98</td><td>34.72</td><td>45.87</td><td>26.53</td><td>21.29</td><td>24.43</td><td>23.82</td></tr><tr><td>GPT-5.6 Sol</td><td>35.79</td><td>38.04</td><td>31.50</td><td>35.12</td><td>21.81</td><td>20.55</td><td>26.11</td><td>25.27</td></tr><tr><td>InternVL3.5-241B</td><td>41.64</td><td>32.31</td><td>32.05</td><td>41.18</td><td>21.72</td><td>22.29</td><td>17.22</td><td>13.12</td></tr><tr><td>DeepSeek-V4</td><td>40.22</td><td>34.38</td><td>32.17</td><td>32.38</td><td>19.25</td><td>21.72</td><td>24.16</td><td>18.21</td></tr><tr><td>GPT-5.6 Terra</td><td>37.60</td><td>34.95</td><td>35.44</td><td>41.17</td><td>19.96</td><td>20.57</td><td>19.65</td><td>22.23</td></tr><tr><td>Gemma 4-12B</td><td>35.49</td><td>45.32</td><td>55.32</td><td>71.06</td><td>23.19</td><td>18.42</td><td>11.02</td><td>11.04</td></tr><tr><td>Qwen3.5-4B</td><td>36.90</td><td>38.48</td><td>40.61</td><td>57.80</td><td>21.03</td><td>16.88</td><td>19.88</td><td>17.24</td></tr><tr><td>InternVL3.5-30B</td><td>42.41</td><td>36.40</td><td>31.11</td><td>46.79</td><td>18.12</td><td>18.63</td><td>15.15</td><td>12.73</td></tr><tr><td>MiniMax-M3</td><td>40.60</td><td>37.80</td><td>34.21</td><td>44.54</td><td>17.22</td><td>17.45</td><td>22.12</td><td>14.17</td></tr><tr><td>GPT-5.6 Luna</td><td>38.82</td><td>41.64</td><td>45.85</td><td>58.41</td><td>15.13</td><td>17.39</td><td>22.60</td><td>16.42</td></tr><tr><td>InternVL3.5-38B</td><td>40.58</td><td>38.85</td><td>37.12</td><td>44.19</td><td>15.90</td><td>15.25</td><td>16.02</td><td>15.48</td></tr><tr><td>InternVL3.5-14B</td><td>39.21</td><td>39.05</td><td>40.11</td><td>49.94</td><td>16.58</td><td>14.37</td><td>16.34</td><td>13.32</td></tr><tr><td>Molmo2-4B</td><td>43.20</td><td>36.65</td><td>30.22</td><td>35.96</td><td>13.28</td><td>14.20</td><td>22.50</td><td>21.11</td></tr><tr><td>Ovis2.6-80B</td><td>45.67</td><td>35.02</td><td>36.34</td><td>44.83</td><td>13.82</td><td>15.61</td><td>17.15</td><td>13.55</td></tr><tr><td>MiniCPM-V 4.5</td><td>44.47</td><td>34.79</td><td>30.45</td><td>46.91</td><td>13.43</td><td>15.67</td><td>18.13</td><td>13.67</td></tr><tr><td>Cosmos3-Super</td><td>39.58</td><td>35.33</td><td>33.04</td><td>44.82</td><td>16.36</td><td>13.82</td><td>17.60</td><td>10.24</td></tr><tr><td>InternVL3.5-8B</td><td>39.68</td><td>39.93</td><td>43.33</td><td>60.74</td><td>18.29</td><td>12.45</td><td>12.02</td><td>9.93</td></tr><tr><td>Molmo2-8B</td><td>41.66</td><td>35.27</td><td>30.21</td><td>39.70</td><td>12.80</td><td>11.85</td><td>20.31</td><td>16.72</td></tr><tr><td>ERNIE 4.5 VL</td><td>39.98</td><td>43.10</td><td>39.32</td><td>62.92</td><td>15.54</td><td>12.11</td><td>12.41</td><td>8.71</td></tr><tr><td>MiniCPM-V 4.6</td><td>45.95</td><td>40.04</td><td>43.45</td><td>55.86</td><td>14.84</td><td>11.64</td><td>9.98</td><td>8.37</td></tr><tr><td>Cosmos3-Nano</td><td>43.71</td><td>38.00</td><td>38.79</td><td>53.31</td><td>15.74</td><td>9.18</td><td>14.85</td><td>11.87</td></tr><tr><td>Phi-4 RV</td><td>45.59</td><td>46.36</td><td>46.67</td><td>60.54</td><td>12.99</td><td>10.26</td><td>11.58</td><td>9.52</td></tr><tr><td>LLaVA-OV2-8B</td><td>38.17</td><td>42.89</td><td>44.14</td><td>58.90</td><td>12.66</td><td>8.58</td><td>12.80</td><td>11.30</td></tr><tr><td>InternVL3.5-4B</td><td>47.72</td><td>47.29</td><td>49.68</td><td>63.10</td><td>12.43</td><td>7.41</td><td>11.00</td><td>8.87</td></tr><tr><td>Cosmos3-Edge</td><td>44.14</td><td>41.70</td><td>43.70</td><td>53.39</td><td>12.34</td><td>7.20</td><td>11.32</td><td>6.94</td></tr><tr><td>Granite Vision-4B</td><td>45.33</td><td>41.91</td><td>48.83</td><td>65.31</td><td>10.70</td><td>7.76</td><td>8.61</td><td>9.97</td></tr><tr><td>Mistral Small 4-119B</td><td>50.91</td><td>49.36</td><td>61.22</td><td>82.80</td><td>6.36</td><td>6.58</td><td>3.50</td><td>1.16</td></tr><tr><td>Gemma 4-E2B</td><td>51.67</td><td>54.13</td><td>59.78</td><td>78.89</td><td>6.14</td><td>4.39</td><td>4.64</td><td>1.03</td></tr><tr><td>Ovis2.6-30B</td><td>67.45</td><td>68.51</td><td>66.22</td><td>53.05</td><td>2.74</td><td>3.94</td><td>7.62</td><td>4.38</td></tr><tr><td>Gemma 4-E4B</td><td>59.68</td><td>63.96</td><td>71.50</td><td>81.94</td><td>4.02</td><td>2.62</td><td>1.57</td><td>1.16</td></tr><tr><td>GPT-6 Astra</td><td>33.39</td><td>38.28</td><td>42.15</td><td>49.65</td><td>21.97</td><td>19.67</td><td>22.14</td><td>21.27</td></tr></table>

Table 6: Identity correspondence within and across image-formation modes. Both conditions are reweighted to the same appearance-multiplicity distribution. Positive $\Delta$ values indicate higher error or Score in the cross-mode condition.
<table><tr><td rowspan="2">Model</td><td colspan="3">Correspondence NAE↓</td><td colspan="3">Score↑</td></tr><tr><td>Within</td><td>Cross</td><td>Δ</td><td>Within</td><td>Cross</td><td> $\Delta$ </td></tr><tr><td>Gemini 3.7 Flash</td><td>30.08</td><td>35.65</td><td>+5.57</td><td>29.73</td><td>34.51</td><td>+4.78</td></tr><tr><td>Muse-Glimmer-30B</td><td>32.81</td><td>42.67</td><td>+9.86</td><td>23.87</td><td>30.54</td><td>+6.67</td></tr><tr><td>Qwen3.8-27B</td><td>24.17</td><td>45.03</td><td>+20.86</td><td>27.14</td><td>28.54</td><td>+1.40</td></tr><tr><td>Qwen3.5-397B</td><td>39.34</td><td>45.02</td><td>+5.68</td><td>24.03</td><td>28.49</td><td>+4.46</td></tr><tr><td>Gemma 4-31B</td><td>41.23</td><td>42.66</td><td>+1.43</td><td>16.34</td><td>25.06</td><td>+8.72</td></tr><tr><td>Qwen3.5-27B</td><td>25.29</td><td>48.37</td><td>+23.08</td><td>24.88</td><td>25.66</td><td>+0.78</td></tr><tr><td>Qwen3.5-122B</td><td>43.87</td><td>48.80</td><td>+4.92</td><td>24.00</td><td>25.40</td><td>+1.39</td></tr><tr><td>Qwen3.5-35B</td><td>38.86</td><td>51.29</td><td>+12.43</td><td>17.86</td><td>24.29</td><td>+6.43</td></tr><tr><td>Gemma 4-26B-A4B</td><td>48.94</td><td>46.93</td><td>-2.01</td><td>13.30</td><td>21.55</td><td>+8.25</td></tr><tr><td>Qwen3.5-9B</td><td>42.81</td><td>53.54</td><td>+10.73</td><td>22.88</td><td>22.18</td><td>-0.70</td></tr><tr><td>GPT-5.6 Sol</td><td>34.72</td><td>50.73</td><td>+16.00</td><td>22.16</td><td>21.86</td><td>-0.29</td></tr><tr><td>InternVL3.5-241B</td><td>49.46</td><td>45.81</td><td>-3.65</td><td>9.58</td><td>21.22</td><td>+11.64</td></tr><tr><td>DeepSeek-V4</td><td>40.29</td><td>45.96</td><td>+5.67</td><td>12.24</td><td>22.10</td><td>+9.86</td></tr><tr><td>GPT-5.6 Terra</td><td>36.27</td><td>50.22</td><td>+13.95</td><td>18.23</td><td>20.95</td><td>+2.73</td></tr><tr><td>Gemma 4-12B</td><td>47.22</td><td>48.19</td><td>+0.97</td><td>9.42</td><td>17.15</td><td>+7.73</td></tr><tr><td>Qwen3.5-4B</td><td>29.29</td><td>57.06</td><td>+27.78</td><td>13.42</td><td>17.56</td><td>+4.14</td></tr><tr><td>InternVL3.5-30B</td><td>50.01</td><td>50.08</td><td>+0.06</td><td>8.60</td><td>18.04</td><td>+9.44</td></tr><tr><td>MiniMax-M3</td><td>46.66</td><td>50.55</td><td>+3.89</td><td>10.44</td><td>17.93</td><td>+7.49</td></tr><tr><td>GPT-5.6 Luna</td><td>31.79</td><td>53.71</td><td>+21.93</td><td>19.81</td><td>17.94</td><td>-1.87</td></tr><tr><td>InternVL3.5-38B</td><td>55.73</td><td>54.91</td><td>-0.82</td><td>8.39</td><td>15.77</td><td>+7.38</td></tr><tr><td>InternVL3.5-14B</td><td>54.41</td><td>53.93</td><td>-0.49</td><td>12.03</td><td>14.69</td><td>+2.66</td></tr><tr><td>Molmo2-4B</td><td>48.83</td><td>62.08</td><td>+13.25</td><td>10.42</td><td>16.25</td><td>+5.82</td></tr><tr><td>Ovis2.6-80B</td><td>57.05</td><td>56.83</td><td>-0.23</td><td>4.91</td><td>16.07</td><td>+11.16</td></tr><tr><td>MiniCPM-V 4.5</td><td>45.79</td><td>55.81</td><td>+10.03</td><td>7.70</td><td>16.15</td><td>+8.44</td></tr><tr><td>Cosmos3-Super</td><td>55.86</td><td>63.83</td><td>+7.97</td><td>9.89</td><td>14.07</td><td>+4.18</td></tr><tr><td>InternVL3.5-8B</td><td>58.65</td><td>57.90</td><td>-0.75</td><td>7.28</td><td>12.32</td><td>+5.04</td></tr><tr><td>Molmo2-8B</td><td>52.57</td><td>67.10</td><td>+14.53</td><td>10.02</td><td>13.58</td><td>+3.56</td></tr><tr><td>ERNIE 4.5 VL</td><td>47.78</td><td>58.75</td><td>+10.97</td><td>14.05</td><td>11.74</td><td>-2.31</td></tr><tr><td>MiniCPM-V 4.6</td><td>47.23</td><td>56.27</td><td>+9.04</td><td>11.44</td><td>11.15</td><td>-0.29</td></tr><tr><td>Cosmos3-Nano</td><td>64.89</td><td>68.99</td><td>+4.10</td><td>8.38</td><td>10.27</td><td>+1.89</td></tr><tr><td>Phi-4 RV</td><td>57.23</td><td>61.23</td><td>+4.00</td><td>8.58</td><td>10.46</td><td>+1.89</td></tr><tr><td>LLaVA-OV2-8B</td><td>47.27</td><td>68.02</td><td>+20.76</td><td>14.56</td><td>9.27</td><td>-5.29</td></tr><tr><td>InternVL3.5-4B</td><td>57.90</td><td>64.63</td><td>+6.73</td><td>6.10</td><td>8.10</td><td>+2.00</td></tr><tr><td>Cosmos3-Edge</td><td>65.15</td><td>66.78</td><td>+1.63</td><td>3.19</td><td>7.85</td><td>+4.66</td></tr><tr><td>Granite Vision-4B</td><td>53.60</td><td>67.84</td><td>+14.24</td><td>6.08</td><td>8.26</td><td>+2.18</td></tr><tr><td>Mistral Small 4-119B</td><td>67.90</td><td>67.83</td><td>-0.07</td><td>3.00</td><td>5.69</td><td>+2.69</td></tr><tr><td>Gemma 4-E2B</td><td>77.49</td><td>65.00</td><td>-12.48</td><td>0.57</td><td>4.16</td><td>+3.60</td></tr><tr><td>Ovis2.6-30B</td><td>53.26</td><td>59.32</td><td>+6.06</td><td>2.15</td><td>4.54</td><td>+2.39</td></tr><tr><td>Gemma 4-E4B</td><td>79.17</td><td>67.43</td><td>-11.74</td><td>0.93</td><td>2.40</td><td>+1.47</td></tr><tr><td>GPT-6 Astra</td><td>30.07</td><td>46.82</td><td>+16.74</td><td>24.67</td><td>20.11</td><td>-4.55</td></tr></table>

## D ERROR PATHS IN MODEL-GENERATED REASONING

A wrong integer does not identify the operation that failed. We therefore examine three cases whose generated reasoning exhibits a different error path: constructing the appearance inventory, linking appearances into physical identities, and applying the requested set operation. For each case, we reproduce the complete reasoning of two representative models in its original order, followed by an analysis of the first consequential divergence. These cases illustrate observable reasoning paths rather than estimate their prevalence, and generated reasoning is not treated as a direct readout of a model’s internal computation.

## D.1 A REFLECTION NEED NOT HAVE A VISIBLE SOURCE

Question. Counting both the potted plants physically present on the shelving unit on the left and the reflections ofpotted plants in the round mirror on the right, how many potted plant appearances are there in the image in total? The image contains two direct appearances and three reflected appearances, giving five appearances in total. One reflected plant has no direct counterpart inside the crop.

Original  
Appearance-to-entityannotation  
![](images/97b82783bb2c49eee97c3e3f28fbe7067a6605cca0373517d1a31bf026147440.jpg)  
D: directly visible physical objectR: reflected appearanceT: through-glass physical object E1, E2, ... denote physical-entity identities; white links connect appearances of the same entity

Figure 13: An appearance inventory that is not pairwise balanced. The crop contains two directly visible plants but three plant reflections. The unmatched reflection remains an observed appearance even though its physical source is outside the direct view.

## google/gemma-4-31B-it

Prediction: 5 CORRECT

## MODEL REASONING

Okay, let’s count the potted plant appearances in the image.

on the same shelf: ‘{"point": [348, 208], "label": "potted plant physically present"}‘. 3. Now, let’s look at the reflections in the mirror. I see a reflection of a potted plant here: ‘{"point": [397, 660], "label": "reflection of a potted plant"}‘. 4. And another one next to it in the mirror : ‘{"point": [417, 758], "label": "reflection of a potted plant"}‘. 5. And one more further to the right in the mirror: ‘{"point": [434, 834], "label": "reflection of a potted plant"}‘.

So, there are 2 physically present and 3 reflections. That’s a total of 5.

## FINAL ANSWER

5

## openbmb/MiniCPM-V-4.6-Thinking

Prediction: 4 INCORRECT

## MODEL REASONING AND FINAL ANSWER

Got it, let’s count. On the left shelving unit, there are two potted plants: one cactus and one black pot with a plant? Wait, no, looking at the left shelf: there’s a black pot with a cactus, and another black pot next to it? Wait, the left shelf has two pots: one with a cactus (potted plant) and one black pot (empty? Or does it have a plant? The image shows two pots on the left shelf: one with a cactus (that’s one), and the other black pot is empty? Wait, no, the first pot (leftmost) has a cactus, then the black pot next to it has a small black pot with a plant? Wait, maybe there are two potted plants on the left shelf: the cactus in the first pot and the small plant in the second black pot? Let me check. The left shelf has two pots: one with a cactus (green, spiky), and another black pot in front with a small plant? Or is that just a pot? Hmm, maybe only one potted plant is visibly on the left shelf (the cactus), and then in the mirror, there are reflections of potted plants. The mirror shows reflections of the hanging lights and maybe the shelves, but the potted plants reflected: the shelf on the right has potted plants, including maybe one small plant reflected. Let’s list:

Physical potted plants on left shelf: the cactus (1) and maybe the small plant in the black pot? If that small black pot has a small plant, that’s 2.

Reflections in mirror: the mirror shows reflections of the hanging lights and possibly the potted plants on the shelf. Looking at the mirror, there are reflections of the potted plants on the shelf (the ones on the left shelf are reflected, and on the right shelf there are potted plants reflected). But the question says "potted plants physically present on the shelving unit on the left" plus "reflections of potted plants in the round mirror".

So physical on left shelf: let’s count the actual plants in the left shelf. Left shelf has two pots : one with a cactus (plant), and another pot (black) which seems to have a small plant? Maybe that s a second plant. So that’s 2 physical plants.

Then reflections in mirror: the mirror reflects the left shelf, so the potted plants there (the cactus and maybe others) are reflected. But the cactus is physical, its reflection? Or the reflected image includes the plant. But the question says "reflections of potted plants in the round mirror"--so how many are visible as reflected plants? Looking at the mirror, we can see reflections of the hanging lights and also reflections of the shelf with plants, including the small plant in the black pot on the left shelf? Maybe there are at least two reflected plants: the cactus (physical, counted once) and its reflection, plus another plant? This is getting complicated

Alternatively, count each potted plant that is either physically present (on the left shelf) or its reflection (in the mirror).

Left shelf: physical potted plants: the cactus (1) and the small plant in the black pot (2)--so 2.

Reflections in mirror: the mirror shows reflections of those two, so that’s 2 more? But wait, the cactus is physical, its reflection is in the mirror. Similarly, the small plant in the black pot on left shelf has a reflection in the mirror. So total appearances: physical plants (left shelf: 2) + reflected plants (mirror: 2, which are the reflections of those 2) -- but that would be 4? But no, the physical plants on left shelf are 2 (cactus and small plant), their reflections in mirror would be two more, so total 4? But that seems off.

Wait, let’s list all visible potted plants:

1. Physical on left shelf: cactus (leftmost pot)--1 2. Black pot next to cactus: small plant? (   
second pot)--1 (total 2) 3. Reflections in mirror: the small plant in the black pot on left shelf   
is reflected, so that’s the same plant? Or two plants? The mirror shows reflections of the shelf,   
including the small plant (so that’s one reflection of that plant), and maybe the cactus reflected?   
The mirror reflects the entire scene, so the cactus’s reflection would be visible, and the small   
plant’s reflection.   
But the question is: "how many potted plant appearances are there in the image in total?"--that   
means count every instance of a potted plant, whether physically present or reflected.   
So physically present: left shelf has 2 potted plants (cactus and the small plant in black pot)--   
2.   
Reflected in mirror: those same 2 potted plants are reflected, so that’s 2 more (the reflections   
count as appearances? Or do we count reflections as separate appearances? If a physical plant   
appears in the mirror, that’s a reflection, which is still an appearance of that plant. So each   
physical plant has one appearance (on shelf) and one appearance (reflected in mirror). So if there   
are 2 physical plants, each has 1 appearance on shelf and 1 in mirror, total 4? But that doesn’t   
make sense.   
Alternatively, count the actual plants: left shelf has 2 potted plants (cactus and small plant)--   
2.   
Mirror reflects: those 2 plants are reflected, so in the mirror, we see those 2 plants (reflected),   
which are the same as the physical ones, so total appearances: physical ones (2) + reflected ones   
(2) = 4? But that would be 4.   
Wait, let’s look at the image: left shelf has two pots: one with cactus, one with small plant (   
black pot). The mirror has a round frame, and reflections show those two pots with plants (so the   
plants are reflected, meaning they are present in the reflection, so they are "appearing" in the   
mirror as plants). So the physical plants are 2, and their reflections are 2, so total appearances   
(physical + reflected) is 4? But that would be 2 physical + 2 reflected = 4.   
But wait, the cactus is one physical plant, its reflection is a second appearance. The small plant   
in the black pot is another physical plant, its reflection is a third appearance. So total   
appearances: 4 (physical: 2, reflected: 2).   
But physical plants on left shelf: 2 (cactus and small plant)--physical appearances: 2.   
Reflections in mirror: the mirror shows 2 reflections (one for each physical plant), so reflected   
appearances: 2.   
Total: 2 + 2 = 4.   
Yes, that makes sense. So total appearances (physical present + reflected in mirror) are 4. </think   
>

Analysis. Gemma’s generated reasoning builds the answer from observations: it enumerates two direct plants, independently enumerates three reflections, and only then adds the two sets. MiniCPM’s rationale reverses that dependency. After finding two direct plants, it assumes that the mirror must contain one copy of each and fixes the reflected count at two. The third reflected plant disappears from the final tally because it violates this assumed one-to-one mapping.

The recorded mistake comes from treating a reflection as a duplicate of the visible crop rather than as another view of the surrounding scene. A mirror may reveal a source outside the camera frustum, omit a visible source because of its orientation, or show one source in several panels. None of these cases preserves the direct count. In this example, the reflected regions must remain in the appearance inventory before their physical sources are considered; the MiniCPM rationale instead uses a one-to-one identity assumption to revise the tally.

## D.2 ENTITY COUNTING DEPENDS ON THE IDENTITY EDGES

Question. How many different shoes are actually in the scene captured in this photo? Four shoe appearances form two direct–reflection pairs. The answer is two physical shoes.

## Original

![](images/73a09e964ceeca3b9c172a79f8b0bf0e0fdaa042200e41a267dc19e3e095e2eb.jpg)

Appearance-to-entity annotation  
![](images/66cf7aaaa0fe2b2439055faeb74e394c9a4bef3cabef8318b92ef2611a99d3f1.jpg)  
D: directly visible physical objectR: reflected appearanceT: through-glass physical object E1, E2, ... denote physical-entity identities; white links connect appearances of the same entity.

Figure 14: Four shoe appearances connected into two identities. The mirror boundary does not remove visual evidence; it determines which observations should be linked before the entities are counted.

## google/gemma-4-31B-it

Prediction: 3 INCORRECT

## MODEL REASONING

Okay, let’s count the shoes in the image.

1. First, there’s a pink high-heeled sandal on the left: ‘{"point": [684, 435], "label": "pink high-heeled sandal"}‘. 2. Next to it is another pink high-heeled sandal in the middle: ‘{"point": [727, 621], "label": "pink high-heeled sandal"}‘. 3. And then there’s a third pink high-heeled sandal lying on its side to the right: ‘{"point": [551, 816], "label": "pink high-heeled sandal"}‘.

So, I count 3 shoes in total.

## FINAL ANSWER

3

## OpenGVLab/InternVL3 5-241B-A28B

Prediction: 2 CORRECT

## MODEL REASONING

Okay, let’s see. The image shows a fluffy gray cat lying on the floor next to a pair of bright pink high-heeled shoes. The shoes are placed in front of a mirror, which is reflecting the scene. The cat’s reflection is also visible in the mirror.

The question is asking how many different shoes are actually in the scene. So, the key here is to determine how many physical shoes are present, not counting the reflections. The mirror shows a reflection of the shoes, which might make it look like there are two pairs, but in reality, there’s only one pair in front of the mirror.

Looking at the image, the shoes in front of the mirror are a pair, so that’s two shoes. The mirror reflects them, making it appear as if there’s another pair behind, but that’s just a reflection. So the actual number of different shoes (i.e., individual shoes) present in the scene is two. However , the question might be phrased to ask for the number of pairs, but the wording says "different shoes," which could mean individual shoes. Wait, the wording says "how many different shoes are actually in the scene." So each shoe is a separate one. But since they are a pair, they are two individual shoes. But maybe the answer is one pair, but the question says "different shoes," so maybe two. Alternatively, maybe the reflection is considered not part of the actual scene. So the

actual shoes are two, one left and one right. The mirror shows their reflection, but those aren’t real. So the answer is two. Wait, but the image shows the shoes in front of the mirror. The mirror reflects them, but the actual shoes are two. So the answer is two. But maybe the user is tricking us into thinking that the reflection is a different shoe. But no, the actual shoes are two. So the integer answer is 2. Wait, but looking at the image again, the shoes are placed in front of the mirror. So the mirror shows a reflection of the shoes, but the actual shoes are two. So total different shoes (individual) is two. So the answer is 2. Alternatively, maybe the question is a trick, but I think the answer is two. Wait, but the image shows two shoes (a pair) and their reflection. So actual shoes are two. So the answer is 2. But maybe the question is phrased as " different" in terms of pairs. Hmm. Alternatively, maybe the shoes are the same, so "different" might refer to unique shoes. But they are a pair, so same design. But the question is about how many different shoes are present. If "different" refers to individual shoes, then two. If it’s asking how many pairs, then one. But the question says "different shoes" and "exactly one integer." Let me check the image again. The shoes are a pair of pink high heels. So two shoes. The mirror reflects them, but that’s not real. So the answer is 2. So the integer is 2. But wait, maybe the user is asking how many unique shoes, like left and right. But they are a pair, so two. So answer 2. Alternatively, maybe the answer is 1, considering a pair as one item. But the question says " shoes" plural. Hmm. Let me think again. The question says "how many different shoes are actually in the scene captured in this photo?" So each shoe is a separate entity. So two shoes. The mirror image is not a real shoe. So answer 2. So the integer is 2. But maybe I’m misunderstanding. Let me check the image again. The shoes are placed in front of a mirror, so the mirror shows their reflection. So in the photo, you can see two shoes and their reflection. But the actual physical shoes are two. So the answer is two. So the integer is 2. Alternatively, if the question is a trick , but I think that’s correct.

## FINAL ANSWER

2

Analysis. Gemma’s rationale stops after detecting three salient shoe regions. InternVL instead explains why the regions are repeated: the shoes lie against a mirror, and the rear-looking pair is generated by reflection. The observable difference is not simply more careful arithmetic. InternVL changes the unit being counted from image regions to connected identity groups.

This distinction is easy to lose because visual similarity alone does not determine identity. Matching color and design are compatible with both a reflected shoe and another shoe from the same product line. The relevant evidence is relational: the mirror boundary, reversed placement, and the alignment between direct and reflected regions. InternVL’s rationale uses these cues to introduce two cross-boundary identity links, whereas Gemma’s does not. Expressed in the benchmark’s annotation graph, their stated interpretations disagree on the edges and therefore on the number of connected components.

## D.3 PHYSICAL-ONLY IS A SET DIFFERENCE, NOT A PHYSICAL COUNT

Question. How many people appear only in the physical space of the stage, with no reflection in the mirror? The directly visible dancer also has reflected appearances. The physical-only set is empty, so the answer is zero.

## google/gemma-4-31B-it

Prediction: 0 CORRECT

## MODEL REASONING

Okay, let’s look at the image to find how many people are in the physical space of the stage but <sub>\*</sub> not reflected in the mirror.

1. First, I identify the people. I see one main person in a red tank top and dark blue pants, leaping across the stage. 2. Then, I see reflections in the mirror behind him. There are two reflections of a person in a red tank top and dark blue pants. 3. Wait, looking closer at the reflections: there’s one reflection that corresponds directly to the main person’s position and pose (the one closest to him). 4. There’s another reflection further back, which seems to be of another person who might be off-camera or further back on the stage. 5. Wait, let’s re-examine. There’s one person physically on the stage (the main subject). 6. There are two reflections. One is a clear reflection of the person jumping. The other reflection is further to the left, and it

![](images/7663aa9b07224855db9a628564afe180be34cffb80cf61a30e23dfe954b3bfc8.jpg)

Appearance-to-entityannotation  
![](images/3c89ef4971aa3ea81cc0ffa622cd0215f62d119b4942223817d761ea48970ad9.jpg)  
D: directly visible physical objectR: reflected appearanceT: through-glass physical object  
E1, E2, ... denote physical-entity identities; white links connect appearances of the same entity

Figure 15: A nonempty physical set but an empty physical-only set. The direct dancer and the two mirror appearances belong to one identity; the exclusion condition removes that identity from the answer.

doesn’t seem to have a corresponding physical person visible in the frame. 7. However, the question asks for "people [who] appear <sub>\*</sub>only<sub>\*</sub> in the physical space of the stage, with <sub>\*</sub>no<sub>\*</sub> reflection in the mirror". 8. The main person on stage <sub>\*</sub>is<sub>\*</sub> reflected in the mirror. 9. Are there any other people on the stage who are not reflected? No, I only see one physical person. 10. Let ’s double check. The erson lea in is clearl reflected. There’s another fi ure in the reflection, but there’s no physical person corresponding to them in the visible part of the stage. 11. So, the only person on the stage has a reflection. Therefore, there are zero people who appear only on the stage with no reflection.

Let me re-read: "How many people appear only in the physical space of the stage, with no reflection in the mirror?" - Person A (leaping): Appears on stage AND has a reflection. - Person B (reflected ): Appears in the reflection, but not on the physical stage in the image.

So, none of the people visible on the stage are without a reflection. Wait, let me look again. Maybe there’s someone else? Looking at the whole image... I see one main person. I see two reflections. One reflection is definitely him. The other reflection is another person in a red top. Where is that person physically? They are not visible in the non-mirrored part of the image. So, the main person is reflected. There’s no one else on the stage who is not reflected.

Therefore, the number is 0.

## FINAL ANSWER

0

## Qwen/Qwen3.8-27B

Prediction: 1 INCORRECT

## MODEL REASONING

The user wants a count of people who appear only in the physical stage space, not as mirror reflections.

Looking at the image, I see a stage with a mirror or reflective surface. There’s a main dancer in the foreground wearing a red tank top and dark blue pants, leaping with arms extended.

Behind him, in what appears to be a mirror reflection, there are two figures also in red tops and dark pants, mirroring the pose. Wait, let me look carefully.

The image shows a dancer in the foreground (the main subject). Behind him, there’s a vertical line that looks like a mirror edge. To the left of that line, there are two reflected figures? Let me look.

Actually, the setup: there’s a mirror on the back wall. The dancer in the foreground is the real person. The reflections in the mirror show his reflection. But I see two figures in the background on the left side.

Let me parse the geometry. The foreground dancer’s body is oriented with his left leg extended to the right (his right leg?), his torso facing us, arms out.

In the background left, I see one figure with arms out (red top, dark pants) and another figure partially overlapping. Hmm.

Let’s look at the reflections. The mirror is likely the dark vertical surface behind. The reflection of the dancer would appear in the mirror. The dancer is leaping to the right (his left). The reflection would be reversed.

In the background, I see two red-topped figures. One is more complete on the left (with arms out to the left), and one is in the middle-left (the one whose body aligns with the foreground dancer’s torso). Actually the middle-left figure looks like a reflection of the foreground dancer’s upper body, and the leftmost figure is the reflection of the lower body? That doesn’t make sense.

Let’s think about mirror geometry. Suppose there are two mirrors at an angle, producing multiple reflections. Or maybe there’s just one mirror and we see the dancer’s reflection plus a second dancer’s reflection? But the question asks how many people appear ONLY in physical space, i.e., not as reflections. So we need to count the real people on stage who do not have a corresponding reflection? Or count real people, excluding reflections?

The phrasing: "How many people appear only in the physical space of the stage, with no reflection in the mirror?" This means count the people who are physically present on stage (real people), as opposed to reflections. But "with no reflection in the mirror" might mean we should count people that we see as physical (not as mirror images). Hmm, but every physical person would have a reflection in the mirror. So "appear only in the physical space" means we look at the image and identify which figures are physical and which are reflections. The physical ones are the answer.

So the question reduces to: how many real dancers are on stage (as opposed to their mirror images)?

Looking at the image, the foreground dancer is clearly the real one (he’s in focus, lit, in front). The background figures are reflections.

But wait, there are two background figures. If there were only one real dancer, his single reflection in a single flat mirror would produce one image. Why two?

Possibility: There are two mirrors meeting at an angle (like a corner), producing two reflections of the same dancer. Or the background shows two reflections of the same dancer from two different mirror panels.

Look at the vertical line in the middle of the image (around x=620). That’s the edge between two mirror panels perhaps. The right side is dark (no reflection visible there because the angle doesn t show the dancer, or it’s a black curtain). The left side is the mirror showing the reflection.

Hmm, but I see two figures on the left side of that vertical line. Let’s look at their positions. One figure (call it A) is at the far left, arms extended to the left, body leaning, legs togetherish. Another figure (call it B) is just left of center, with one arm extended up-left and torso in red, and a leg extended to the right (the leg at around (430,620)).

Actually, figure B’s leg extended to the right (the dark pant leg ending in a shoe at ˜(480,630)) might be the reflection of the foreground dancer’s extended leg? The foreground dancer’s extended leg goes to the right (to ˜(900,700)). The reflection of that leg in a vertical mirror behind him would go to the left, not right. Hmm.

This is getting too detailed. Let’s think about the intended answer. The question is likely from a VQA dataset testing reflection understanding. The expected answer is probably 1: there is one real dancer on stage; all other figures are reflections.

But the phrase "with no reflection in the mirror" is odd. Maybe it means: count the people such that the person themselves is in physical space (not a reflection). I.e., how many physical people do we see? That’s 1.

But maybe there are two physical dancers, and the mirror shows reflections of both, making it confusing. Let’s look: foreground dancer is real. Is there a second real dancer hidden? The background figures are dimmer and look like reflections (they’re behind the mirror line, less lit). The leftmost figure A and middle figure B could be two reflections of the one real dancer from two mirror surfaces.

Actually, look at the geometry: The real dancer’s body: his right arm (our left) is extended forward-left, his left arm (our right) extended forward-right? No, both his arms are extended to our left actually. Look: his shoulders, one arm goes to our left-up (the hand at ˜(200,120)), the other arm also to our left but more horizontal (hand at ˜(270,110)). Both arms point to our left. His body is twisted. His right leg (our right) is extended to our right. His left leg (our left) is bent, planted? Actually his left leg (our left) is extended down-left to the foot at ˜(160,880). So both legs spread: one to our left-down, one to our right-down. It’s a leap/split jump.

Now a vertical mirror behind him (the dark backdrop) would reflect him. The reflection’s arms would point to the right (mirror flips horizontally). But the background figures have arms pointing to our left as well. Hmm, that suggests they are not simple horizontal flips, or the mirror is angled.

This is too much. The likely intended simple answer: 1.

Let’s consider the possibility the answer is 1 because only the main dancer is "physical" and the rest are reflections. The question’s wording "appear only in the physical space of the stage, with no reflection in the mirror" might be a clumsy way of saying "count the real people (not the mirror images)." So answer = 1.

But wait, maybe the question is the opposite: count people who appear ONLY in physical space, meaning we should look at the set of "people" depicted and determine which ones are not reflections . The reflections are "appear in the mirror." The physical people "appear in physical space." So count physical = 1.

Alternatively, maybe there are 2 physical people: the main dancer and a partner off to the side that we see only physically, whose reflection isn’t visible. But I don’t see a second clear physical person. The two background figures are both in the mirror region (left of the vertical seam). They look like reflections.

Let me look at the lighting: foreground dancer is brightly lit. Background figures are also fairly lit but appear behind a glass line. The vertical line at x˜190 and x˜620 might be the edges of a mirror panel? Actually there’s a vertical line at around x=190 (a thin bright line) and at x=620. These could be the frame of a mirror that occupies the middle region (x from 190 to 620). In that middle region we see figure B (the reflection of the foreground dancer). To the left of x=190, we see figure A, which is on a different mirror panel? Or figure A is the reflection on a side mirror?

Hmm, if the mirror occupies x in [190,620], then figure B is the reflection of the foreground dancer in that mirror. Then what is figure A? Figure A at x in [60,280] partially overlaps the mirror region and partially left of it. Figure A’s body is to the left.

I think the scene is: a single large mirror on the back wall, but the photo captures the dancer’s reflection, and due to the dancer’s pose, the reflection appears as one figure. But we see two figures in the background, which is puzzling.

Let’s look very carefully at the background. I’ll describe pixel regions.

Figure A (leftmost): head at ˜(185,260), red top torso at ˜(180,380), arms extended to upper-left ( hands at ˜(140,275) and ˜(70,340)), legs in dark pants going down-left to foot at ˜(75,750). So figure A is a full-body figure with arms to the left.

Figure B (middle): head at ˜(240,255), red top torso at ˜(240,400), one arm extended to upper-left (hand at ˜(185,300)? or that’s figure A’s?), and a leg extended to the right (the dark pant leg from ˜(350,500) to (480,630)). Figure B’s other leg? Not clearly visible, maybe overlapping with foreground dancer’s leg.

Hmm, so figure B’s extended leg goes to the right (our right). The foreground dancer’s extended leg also goes to our right (to 900,700). If the mirror is behind the dancer, the reflection’s extended leg would go to our left. So figure B’s right-going leg doesn’t match a simple reflection.

Wait, maybe figure B is actually the real second dancer? But figure B is dimmer and behind the   
vertical line? The vertical line at x=190 passes through figure A’s torso. The vertical line at x   
=620 is to the right of figure B.   
I’m overanalyzing a low-res image. Let’s think about typical VQA groundings. This image likely from   
a "reflection" reasoning set. The question: "How many people appear only in the physical space of   
the stage, with no reflection in the mirror?" The expected answer is likely 1.   
But let’s double check by thinking: maybe the trick is that the foreground dancer’s reflection is   
figure B, and figure A is the reflection of a second real dancer who is mostly out of frame? No.   
Alternatively maybe the answer is 0? No, the foreground dancer is physical.   
Let’s think about "appear only in the physical space ... with no reflection in the mirror." Could   
be interpreted as: among all the person-figures visible in the image, how many of them are the kind   
that exist only physically (i.e., are real people) and not also present as a reflection? But a   
real person also appears as a reflection, so they appear in both. The phrase "with no reflection in   
the mirror" might modify "appear only in the physical space" meaning we consider the physical  
space appearances that don’t have a corresponding mirror appearance? That’s weird.   
Maybe simpler: The image contains person-figures. Some are real (on stage), some are reflections (   
in mirror). The question asks how many of the figures are real (physical). The reflection figures   
are "in the mirror." The real figures are "in physical space." So count real = ?   
If there’s one real dancer and his reflection produces the background figures, then real count = 1.   
But why two background figures then? Possibly because there are two mirrors (the stage has a   
mirrored wall at an angle, like a corner reflector), producing two reflections of the one dancer.   
That’s common in dance sets to create multiple images. So one dancer -> one physical + two   
reflections = 3 figures total. The question: physical only = 1.   
Yes I think that’s it. The set designer uses angled mirrors to multiply the dancer’s image. So we   
see 3 figures: 1 real, 2 reflections.   
Thus answer = 1.   
Let’s sanity check the visual: real dancer in front, bright. Two dimmer figures behind, consistent   
with reflections. Yes.   
Thus the number of people appearing only in physical space = 1.   
The question wants exactly one integer. So output 1.   
I’m fairly confident. Let’s also consider if maybe the foreground is the reflection and the   
background is real? No, foreground is clearly in front, on the stage floor with shadow, bright.   
Background is in the dark mirror area.   
Thus answer 1.   
I’ll output 1.

## FINAL ANSWER

Analysis. Qwen’s rationale describes the relevant visual structure: one foreground dancer, background reflections, and multiple mirror panels as the source of repetition. The answer changes after the question is paraphrased as the number of “real dancers on stage.” The stated computation then becomes |P| although the question asks for |P \ R|, where P contains identities with direct appearances and R contains identities with reflected appearances.

Gemma’s rationale retains the exclusion predicate. It first places the directly visible dancer in P, then places the same identity in R, and removes it. The answer is zero even though the physical set is nonempty. The two recorded rationales therefore differ at the requested operation: Qwen’s scene description is largely sufficient, but its final count is not licensed by that description.

The unsuccessful trace repeatedly strengthens the correct claim that there is one physical dancer, yet never restores the discarded phrase “with no reflection.” In this case, additional reasoning increases confidence without improving semantic fidelity. The missing operation, rather than the amount of visual inspection, determines the wrong answer.

Implication for evaluation. The cases establish a narrower but important point: the same final counting error can accompany different observable reasoning failures. In the plant rationale, a oneto-one reflection assumption changes the appearance tally; in the shoe rationale, the stated identity links are incomplete; in the dancer rationale, the exclusion operator is dropped after the scene has been described. Aggregate integer metrics identify which question families and scene structures are difficult, but they cannot distinguish these paths on their own.

The joint-presentation result in Section 4.3 is compatible with reuse of appearance and identity information across related questions, but it does not identify that reuse as the causal internal mechanism. Directly separating appearance detection, identity linking, and set operations would require structured predictions for the appearance nodes, formation labels, identity edges, and requested operation. The present traces serve instead as qualitative examples of why those intermediate outputs would make future diagnosis more precise.