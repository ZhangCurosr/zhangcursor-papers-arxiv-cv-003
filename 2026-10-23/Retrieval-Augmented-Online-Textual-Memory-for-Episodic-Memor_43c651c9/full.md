# Retrieval-Augmented Online Textual Memory for Episodic Memory Video Question Answering

Raffaele Cal\`ı<sup>1</sup>, Giuseppe Lando<sup>1</sup>, Rosario Forte<sup>1</sup>, Antonino Furnari<sup>1</sup>

<sup>1</sup> Department of Mathematics and Computer Science, University of Catania, Italy

{raffaele.cali, giuseppe.lando}@studium.unict.it, rosario.forte@phd.unict.it, antonino.furnari@unict.it

Abstract—Long egocentric videos pose a central challenge for multimodal language models: the visual stream must be compressed into a memory representation that can be searched efficiently while still preserving the evidence needed for question answering. Building on an online textual-memory framework in which first-person videos are converted into temporally grounded captions, this work systematically studies the role of retrieval in episodic-memory video question answering. We evaluate how different retrieval units, query formulations, context budgets, and reranking strategies affect multiple-choice answering on QAEgo4D-Closed, isolating retrieval as the main variable while keeping the underlying generated memory fixed. Our best configuration, PARENT, matches fine-grained caption subchunks, then provides their full parent captions to the reasoner. It achieves 56.4% accuracy, matching strong full-memory and retrieval baselines while exposing only a small fraction of the textual memory at question time. More importantly, our analysis shows that better temporal coverage does not necessarily translate into better answers: increasing the retrieval budget improves temporal Hit@K and Recall@K, but can reduce accuracy by introducing visually plausible distractors and uncertain caption evidence. These findings suggest that retrieval-augmented egocentric memory should not be optimized only for evidence recall, but also for the reliability, specificity, and answer-discriminative value of the retrieved context. The study motivates more structured episodic memories that move beyond raw caption retrieval toward evidence-aware representations for long-form egocentric reasoning.

Index Terms—Egocentric Vision, Retrieval-Augmented Generation, Episodic Memory, Video Question Answering.

## I. INTRODUCTION

Egocentric video question answering requires a model to reason over what a camera wearer saw, did, handled, and encountered across long first-person video streams. This setting is closely related to artificial episodic memory, where an agent should recall past events and use them to answer later queries [1]–[3]. It is also central to wearable assistants, lifelogging, and embodied perception, where questions may refer to objects, locations, actions, and state changes observed long before the question is asked [4]–[6]. However, egocentric video exposes a fundamental scalability problem. A single recording may contain many visually similar moments, repeated objects, partial views, camera motion, occlusions, and small state changes. Directly processing the video at question time is expensive, while reading an entire textual memory can still exceed practical context budgets and introduce plausible distractors.

![](images/cd255866c623bd2d598fafa8a4d6e6df3523a3a6a213bf92ffecca6dc32cf85c.jpg)  
Fig. 1: Overview of our retrieval-augmented textual-memory pipeline for egocentric video question answering. The egocentric video stream is processed online by a video descriptor model into captions, which are stored in an online textual memory. At question-answering time, the system uses the question and answer options to retrieve the most relevant subchunks, which are expanded to their parent captions and passed to the reasoner to predict the answer.

Recent work on Online Episodic Memory Video Question Answering (OEM-VQA) addresses this issue by processing the visual stream sequentially and retaining a compact memory that can be queried later, rather than reprocessing the full video for every question [7], [8]. Within this direction, textualmemory approaches convert egocentric clips into language descriptions, yielding memories that are compact, humanreadable, and reusable across multiple queries [4]–[6], [8]. Prior work has shown that such memories can match dedicated state-of-the-art systems on QAEgo4D-Closed while using substantially less memory than approaches that retain visual features or cache large video representations [3], [7], [8]. This makes textual memory an appealing substrate for scalable and inspectable egocentric reasoning.

This paper builds on this textual-memory paradigm and studies a narrower but crucial question: once the visual stream has been compressed into textual memory, which parts of that memory should be provided to the reasoner at query time? As illustrated in Fig. 1, our pipeline separates online memory construction from question-time retrieval and reasoning. Video clips are first converted into captions and stored in an online textual memory. When a question arrives, the system retrieves relevant subchunks and expands them to their parent captions, providing the reasoner with compact but contextual evidence.

Retrieval-augmented generation (RAG) was introduced to equip large language models with explicit non-parametric memory [9]. In episodic video QA, however, retrieval does more than reduce context length, it changes the evidence distribution seen by the reasoner. A retrieval module must recover the relevant event, preserve enough surrounding context to disambiguate the answer options, and avoid introducing visually plausible but incorrect evidence from other moments in the same video.

We therefore isolate retrieval as the main object of study. All configurations operate on the same generated textual memory and differ only in how memory entries are indexed, retrieved, transformed, or passed to the reasoner. We compare flat caption retrieval, subchunk retrieval, parent retrieval, hybrid dense–lexical retrieval, and reranking strategies. Our strongest configuration uses PARENT retrieval: captions are split into fine-grained subchunks for embedding and matching, but the corresponding full parent captions are provided to the reasoner. This strategy reaches 56.4% accuracy on QAEgo4D-Closed, matching strong state-of-the-art baselines while using only 3,600 tokens per query on average.

The main finding of our study is that retrieval quality and answer accuracy do not improve monotonically together. Increasing the number of retrieved captions raises temporal Hit@K and Recall@K, but answer accuracy peaks earlier and then declines. In long egocentric videos, additional retrieved context often contains repeated objects, similar locations, uncertain descriptions, or actions that are semantically close to the question but correspond to the wrong moment. As a result, a reasoner may become less reliable even when the retrieved set covers more of the annotated evidence window. This suggests that episodic-memory QA should not optimize retrieval only for temporal coverage, but also for the specificity and answer-discriminative value of the retrieved evidence.

Our contributions are threefold:

• We provide a systematic retrieval-centred study for textual-memory OEM-VQA, explicitly building on prior training-free memory construction pipelines and isolating retrieval as the main variable.

• We compare multiple retrieval units, context budgets, and ranking strategies on QAEgo4D-Closed, showing that PARENT retrieval provides the best trade-off between fine-grained matching and contextual reasoning.

• We analyze the mismatch between temporal retrieval metrics and final answer accuracy, showing that higher evidence coverage can introduce hard distractors and reduce the reliability of the final reasoner.

## II. RELATED WORK

## A. Egocentric Video Question Answering

Episodic memory in egocentric vision concerns the ability to answer questions about past first-person observations, such as where an object was placed, which action was performed, or what happened before or after a given event [1]–[3]. Ego4D formalized this problem through the Natural Language Queries (NLQ) benchmark, where models must localize the temporal segment containing the evidence needed to answer a naturallanguage query [2]. Early work on episodic-memory QA further studied open-ended question answering over egocentric videos [10], while Ground VQA introduced grounded question answering in long egocentric videos and the QAEgo4D-Closed multiple-choice setting [3]. This formulation reduces the ambiguity of free-form answer evaluation, but still requires models to identify the relevant moment and distinguish the correct answer from plausible distractors.

Recent work has increasingly focused on long-horizon and memory-based egocentric understanding [5]–[8], [11]. MM-Ego studies egocentric multimodal LLMs for video QA and highlights the difficulty of preserving visual details over long first-person videos [11]. Streaming video-language models such as VideoLLM-online address online video understand ing by processing video streams incrementally rather than assuming that the full video can be reprocessed at query time [7]. Complementary textual-memory approaches decouple perception and reasoning by converting observed clips into language descriptions and answering later questions over the accumulated memory [4]–[6], [8]. Our work builds on this textual-memory paradigm, but focuses specifically on retrieval: given an existing temporally grounded textual memory, we study which evidence should be selected and provided to the reasoner at query time.

## B. Video Retrieval-Augmented Generation

Retrieval-augmented generation was introduced as a way to combine parametric language models with explicit nonparametric memory [9]. In video understanding, this idea becomes more challenging because retrieval must also handle temporal grounding and modality compression. DrVideo converts long videos into document-like representations and retrieves relevant textual evidence to support long-video understanding [12]. Video-RAG similarly uses retrieval-augmented mechanisms to improve long-video comprehension by selecting auxiliary visual-textual evidence before generation [13]– [15]. Our work is complementary to these video RAG systems. Rather than training or designing a new video model, we isolate retrieval within an online textual-memory pipeline whose captions have already been generated from the egocentric stream. This allows us to study how retrieval units, query formulation, context budget, and memory format affect downstream multiple-choice answering. In this setting, retrieval is not only a way to reduce context length, it determines the evidence distribution seen by the reasoner and can therefore improve or degrade answer selection depending on the relevance and reliability of the retrieved captions.

## C. Memory Structures for Long Context Reasoning

Textual memory is attractive for long-form egocentric reasoning because it is compact, human-readable, easy to index, and reusable across multiple queries. However, the structure of the memory determines what can be recovered at query time. Long-context memory systems have shown that retrieval can benefit from organizing information beyond independent chunks, for instance through hippocampal-inspired memory and graph-based associations [16], [17].

![](images/fc807aec243a94812751fd4c47784c1266f913a529ad35662f65af0d54671865.jpg)  
Fig. 2: Overview of the proposed retrieval-augmented online textual-memory pipeline. The video stream is processed incrementally by a descriptor model into captions, which are then chunked, embedded, and stored in a vector database. At question-answering time, the query and candidate answers are used to retrieve relevant memory entries, which are provided to the reasoner to predict the final answer.

This issue is especially important in egocentric video, where many questions are not answered by object presence alone, but by temporal and relational context: where an object appeared, when it was manipulated, what action preceded it, or how it related to other objects and locations [3], [18], [19]. For this reason, we analyze how retrieval over caption-based memories succeeds or fails when questions require temporally grounded object, action, and location evidence. Our results suggest that useful memory should preserve entities together with temporal anchors, actions, and locations [18], [20], but also keep the retrieved textual context compact. This indicates that caption-based episodic memories should optimize not only for evidence recall, but also for token efficiency, context specificity, and answer-discriminative value.

## III. METHOD

Our retrieval-augmented formulation builds on the online textual-memory pipeline introduced in [8], where egocentric video clips are processed incrementally into a compact textual memory and later queried by a language-model reasoner. We extend this formulation by indexing the generated captions and retrieving a query-specific subset of memory entries before answering multiple-choice questions. The overall pipeline is shown in Fig. 2. It is organized into three main stages: an Online Memory Constructor, a Memory Manager, and a $Q A$ Handler. Given an incoming video stream, clips are processed incrementally. For each new clip, a multimodal descriptor model produces a natural-language caption. The caption is then passed to a chunking module, which decomposes it into smaller textual chunks, and each chunk is embedded by a text embedding model (Fig. 2(a)). Each chunk is stored with its temporal metadata and embedding in a vector database (Fig. 2(b)). At question-answering time, the input question and the candidate answers are used to form a retrieval query. A context retriever performs similarity search over the vector database and returns a retrieved context. Finally, a language-model reasoner receives the question, the candidate answers, and the retrieved context and predicts the final answer (Fig. 2(c)). This formulation standardizes all memory operations around indexed textual chunks. In contrast to treating a full caption as the only memory unit, our pipeline explicitly separates caption generation, chunk construction, embedding, indexing, retrieval, and reasoning.

## A. Online Memory Constructor

Let V denote an egocentric video stream. The stream is divided into a sequence of temporally ordered clips

$$
V = \{ c _ { 1 } , c _ { 2 } , \ldots \ldots , c _ { N } \} ,\tag{1}
$$

where each clip $c _ { i }$ corresponds to a temporal interval $[ t _ { i } ^ { s } , t _ { i } ^ { e } ]$

When a new clip $c _ { i }$ becomes available, a multimodal descriptor model $D$ generates a natural-language caption:

$$
n _ { i } = D ( c _ { i } ) .\tag{2}
$$

The caption $n _ { i }$ describes the visual content observed in the temporal interval $[ t _ { i } ^ { s } , t _ { i } ^ { e } ]$ . The caption is then passed to a chunking module, which decomposes it into a set of textual chunks:

$$
\{ n _ { i , 1 } , n _ { i , 2 } , \dots , n _ { i , m _ { i } } \} = \mathrm { C h u n k i n g } ( n _ { i } ) ,\tag{3}
$$

where $n _ { i , j }$ denotes the $j \cdot$ -th textual chunk generated from caption $n _ { i } .$ and $m _ { i }$ is the number of chunks produced for clip $c _ { i }$ .

The output of the Online Memory Constructor is therefore a temporally grounded set of textual chunks. Each chunk inherits the temporal interval of the clip from which its parent caption was generated. This allows the memory to preserve temporal information while operating at a finer textual granularity.

## B. Memory Manager

The Memory Manager is responsible for embedding, indexing, and storing the textual chunks produced by the Online Memory Constructor.

Each textual chunk $n _ { i , j }$ is passed through a text embedding model E

$$
e _ { i , j } = E ( n _ { i , j } ) ,\tag{4}
$$

where $e _ { i , j }$ is the dense vector representation of chunk $n _ { i , j }$

The indexer stores each chunk in the vector database as a memory entry:

$$
M _ { i , j } = ( t _ { i } ^ { s } , t _ { i } ^ { e } , n _ { i , j } ; e _ { i , j } ) ,\tag{5}
$$

where $t _ { i } ^ { s }$ and $t _ { i } ^ { e }$ are the start and end timestamps of the corresponding clip, $n _ { i , j }$ is the textual chunk, and $e _ { i , j }$ is the embedding used for indexing and similarity search.

After processing the first i clips, the vector database contains the set of indexed memory entries

$$
\mathcal { M } _ { i } = \{ M _ { \ell , j } : 1 \leq \ell \leq i , 1 \leq j \leq m _ { \ell } \} .\tag{6}
$$

The memory therefore grows incrementally as the video stream is processed. No predefined upper bound on the number of memory entries is required.

For simplicity, when referring to the full memory after the stream has been processed up to the point required by a question, we write

$$
\begin{array} { r } { \mathcal { M } = \{ M _ { i , j } \} _ { i , j } . } \end{array}\tag{7}
$$

## C. QA Handler

At question-answering time, the QA Handler receives a query event consisting of a natural-language question q and a set of candidate answers $\mathcal { A } = \{ a _ { 1 } , a _ { 2 } , \ldots , a _ { L } \}$ . The context retriever builds a retrieval query from the question and from the candidate answers, defined as

$$
r ( q , \mathcal { A } ) = [ q ; a _ { 1 } ; . . . ; a _ { L } ] .\tag{8}
$$

The notation $[ q ; a _ { 1 } ; \ldots ; a _ { L } ]$ denotes the textual concatenation of the question and all answer options.

The retrieval query is embedded using the same embedding model E as $\begin{array} { l } { { e _ { q } } } \end{array} = \begin{array} { l } { { E ( r ( q , \mathcal { A } ) ) } } \end{array}$ , and the context retriever performs similarity search over the vector database. Given a similarity function sim $. ( \cdot , \cdot )$ , the relevance score between the query and a memory entry $M _ { i , j }$ is defined as

$$
s ( r ( q , \mathcal { A } ) , M _ { i , j } ) = \sin ( e _ { q } , e _ { i , j } ) .\tag{9}
$$

The retrieved context is obtained by selecting the top-K memory entries:

$$
\mathcal { C } _ { q } = \mathrm { T o p K } _ { M _ { i , j } \in \mathcal { M } } s ( r ( q , \boldsymbol { A } ) , M _ { i , j } ) .\tag{10}
$$

Each retrieved entry contains both textual evidence and its associated temporal interval, i.e. $\mathcal { C } _ { q } = \{ ( t _ { i } ^ { s } , t _ { i } ^ { e } , n _ { i , j } ) \}$ over the retrieved entries.

Finally, the reasoner model R receives the question, the candidate answers, and the retrieved context, and predicts the final answer:

$$
\hat { a } = R ( q , \mathcal { A } , \mathcal { C } _ { q } ) ,\tag{11}
$$

where ${ \hat { a } } \in A .$

We consider different retrieval strategies (see Fig. 3).

## D. Flat Caption Retrieval

In the flat caption retrieval setting, each full caption $n _ { i }$ is treated as a single searchable unit instead of being decomposed into smaller chunks. In this case, the memory entry associated with clip $c _ { i }$ is

$$
M _ { i } ^ { \mathrm { H a t } } = ( t _ { i } ^ { s } , t _ { i } ^ { e } , n _ { i } ; e _ { i } ) ,\tag{12}
$$

with $e _ { i } = E ( n _ { i } )$ . The retrieved context is obtained by selecting the top-K full-caption entries by similarity:

$$
\mathcal { C } _ { q } ^ { \mathrm { f i a t } } = \mathrm { T o p K } _ { M _ { i } ^ { \mathrm { f l a t } } } \sin ( E ( r ( q , \mathcal { A } ) ) , e _ { i } ) .\tag{13}
$$

This strategy provides the reasoner with complete captions, preserving the full local context generated for each clip.

![](images/420f69c83b597fabed999f77058b246b9edabc64407c62d7cddb0322e9b3fc8d.jpg)  
Fig. 3: The proposed retrieval strategies.

## E. Subchunk Retrieval

Each caption $n _ { i }$ is decomposed into smaller textual chunks $n _ { i , j }$ , and each chunk is independently embedded and indexed as $M _ { i , j }$

Retrieval is performed directly over the chunk-level vector database:

$$
\mathcal { C } _ { q } ^ { \mathrm { s u b } } = \mathrm { T o p K } _ { M _ { i , j } \in \mathcal { M } } \mathrm { s i m } ( E ( r ( q , \boldsymbol { A } ) ) , e _ { i , j } ) .\tag{14}
$$

The retrieved chunk-level context $\mathcal { C } _ { q } ^ { \mathrm { s u b } }$ is then passed directly to the reasoner. This variant uses shorter and more focused evidence snippets as the final retrieved context.

## F. Parent Retrieval

The parent retrieval variant separates the unit used for retrieval from the unit provided to the reasoner. Retrieval is performed over chunk-level entries $M _ { i , j }$ , but each retrieved chunk is mapped back to its parent caption $n _ { i }$

Let $\mathcal { U } _ { q }$ denote the set of top-K<sub>s</sub> retrieved chunk-level entries:

$$
\mathcal { U } _ { q } = \mathrm { T o p K } _ { M _ { i , j } \in \mathcal { M } } \sin ( E ( r ( q , \boldsymbol { A } ) ) , e _ { i , j } ) .\tag{15}
$$

The parent context is obtained by collecting the full captions associated with the retrieved chunks:

$$
\mathcal { C } _ { q } ^ { \mathrm { p a r e n t } } = \mathrm { U n i q u e } \{ ( t _ { i } ^ { s } , t _ { i } ^ { e } , n _ { i } ) : M _ { i , j } \in \mathcal { U } _ { q } \} .\tag{16}
$$

The reasoner then predicts the answer using the expanded parent context $\mathcal { C } _ { q } ^ { \mathrm { p a r e n t } }$ <sup>t</sup>. This strategy uses fine-grained chunks for retrieval while preserving the broader caption context for reasoning.

## G. Hybrid Retrieval

We also evaluate a hybrid retrieval strategy that combines dense semantic retrieval with lexical BM25 retrieval. Dense retrieval ranks memory entries according to embedding similarity, while BM25 ranks them according to lexical matching between the retrieval query and the textual chunks.

Given a memory entry $M _ { i , j }$ , let $\rho _ { \mathrm { d e n s e } } ( r ( q , \mathcal { A } ) , M _ { i , j } )$ denote its rank according to dense retrieval, and let ρ<sub>BM25</sub> $( r ( q , \mathcal { A } ) , M _ { i , j } )$ denote its rank according to BM25. The two rankings are combined using reciprocal rank fusion:

$$
\begin{array} { r l } {  { s _ { \mathrm { R R F } } ( \boldsymbol { r } ( \boldsymbol { q } , \boldsymbol { A } ) , \boldsymbol { M } _ { i , j } ) = \frac { 1 } { k _ { 0 } + \rho _ { \mathrm { d e n s e } } ( \boldsymbol { r } ( \boldsymbol { q } , \boldsymbol { A } ) , \boldsymbol { M } _ { i , j } ) } } \quad } & { } \\ & { + \frac { 1 } { k _ { 0 } + \rho _ { \mathrm { B M 2 5 } } ( \boldsymbol { r } ( \boldsymbol { q } , \boldsymbol { A } ) , \boldsymbol { M } _ { i , j } ) } , } \end{array}\tag{17}
$$

where $k _ { 0 }$ controls the contribution of lower-ranked entries. The hybrid retrieved context is then defined as

$$
\mathcal { C } _ { q } ^ { \mathrm { h y b r i d } } = \mathrm { T o p K } _ { M _ { i , j } \in \mathcal { M } } s _ { \mathrm { R R F } } ( r ( q , \boldsymbol { A } ) , M _ { i , j } ) .\tag{18}
$$

TABLE I: Ablation of the proposed RAG configurations on QAEgo4D-Closed using a fixed retrieval budget of $K = 1 2$ chunks.
<table><tr><td>System</td><td>Acc.</td><td>Hit@K</td><td>Recall@K</td></tr><tr><td>PARENT</td><td>56.4%</td><td>79.0%</td><td>70.7%</td></tr><tr><td>FLAT</td><td>55.6%</td><td>80.4%</td><td>72.3%</td></tr><tr><td>SUB</td><td>54.2%</td><td>76.6%</td><td>67.5%</td></tr><tr><td>PARENT-RERANK</td><td>53.8%</td><td>79.0%</td><td>70.7%</td></tr><tr><td>SUB-RERANK</td><td>53.4%</td><td>82.4%</td><td>74.3%</td></tr><tr><td>HYBRID</td><td>53.0%</td><td>78.8%</td><td>69.1%</td></tr><tr><td>PARENT-TIME</td><td>52.8%</td><td>68.0%</td><td>58.8%</td></tr></table>

![](images/30586aa13a8fe411fb21997dc0fad60c8470d56b6055723018ca0ce5541e3c8e.jpg)  
Fig. 4: Retrieval budget curve for PARENT. Increasing K improves temporal coverage, but answer accuracy peaks at an intermediate retrieval budget.

As in parent retrieval, the fused top-K subchunks are expanded to their parent captions before the reasoner produces the final prediction.

Hybrid retrieval is intended to combine the strengths of dense semantic search and lexical matching.

## H. Evaluation

We report answer accuracy as the percentage of questions for which the predicted answer matches the ground-truth answer. For retrieval diagnostics, we also report temporal Hit@K and Recall@K against the annotated evidence region of each question. Hit@K measures whether at least one of the top-K retrieved memory entries overlaps with the annotated evidence region for the question. Recall@K measures how much of the annotated evidence region is covered by the temporal intervals of the top-K retrieved entries. By construction, Hit@K ≥ Recall@K, since Hit@K only requires a single overlapping entry while Recall@K measures the covered fraction of the evidence window.

Together, these diagnostics evaluate the temporal quality of the retrieval module, while answer accuracy measures the performance of the full retrieval-augmented reasoning pipeline.

## IV. EXPERIMENTAL RESULTS AND ANALYSIS

This section reports the experimental results on the QAEgo4D-Closed [3] test set and analyzes how retrieval design affects both temporal evidence coverage and final answer accuracy. All experiments use Qwen3-VL-8B [21] as the descriptor model and Gemini 2.5 Flash [22] as the reasoner model, with temperature set to 0 and structured JSON decoding. Each video is processed in a streaming setting using non-overlapping clips of 15 seconds.

We use the captioning pipeline and resulting textual memory described in [8], keeping the memory fixed across all experiments. Dense retrieval uses llama-3.2-nemoretriever-300m-embed-v1 [23]. Similarity is computed as cosine similarity, equivalently as dot product over normalized embeddings. The retrieval query is formed by concatenating the question with all answer options. Unless otherwise specified, we use a retrieval budget of $K \ = \ 1 2$ . For PARENT-based methods, we retrieve 36 subchunks and keep up to $K = 1 2$ unique parent chunks as the final context before passing the context to the reasoner to ensure a fair comparison with other methods. HYBRID combines dense retrieval and BM25 using reciprocal rank fusion with $k _ { 0 } = 6 0$ . PARENT-TIME applies a temporal cutoff to model the strictly online setting, retrieving only memory available up to the evidence time, serving as a diagnostic test. Reranking uses nvidia/rerank-qa-mistral-4b [24] before parent expansion: PARENT-RERANK reranks the same candidate pool as PARENT, preserving the selected parents and temporal metric.

## A. Analysis of Retrieval Strategies

Table I reports the ablation of the proposed RAG configurations using a fixed retrieval budget of K = 12 retrieved units. The best result is obtained by PARENT, which reaches 56.4% accuracy, with a Hit@K of 79.0% and a Recall@K of 70.7%. The second-best configuration is FLAT, which achieves a comparable accuracy of 55.6% with slightly higher Hit@K and Recall@K. This suggests that flat retrieval is already competitive when the retrieved captions contain sufficient context, while the parent strategy provides a better balance between precise matching and contextual grounding.

Interestingly, SUB-RERANK obtains the strongest retrieval scores among the tested configurations, with the highest Hit@K and Recall@K. However, this does not translate into the best final accuracy. This gap indicates that retrieving more relevant subchunks is not sufficient by itself, since the reasoner also needs enough surrounding information to correctly interpret the evidence and select the answer. In this sense, retrieval granularity plays a central role. Fine-grained subchunks are useful for locating the relevant moment, but using them alone can fragment the context available to the model. Conversely, parent captions preserve the temporal and semantic context around the matched evidence, leading to more robust answer generation.

![](images/dafd3ccf0da43b9e8738a4aad8297216fc49d852dc9b1c0a044d26a6988539a0.jpg)  
Fig. 5: Qualitative examples of retrieval-augmented generation behavior.

TABLE II: Comparison with state-of-the-art methods on QAEgo4D-Closed. For methods based on MLLMs, we report the number of tokens used at query time.
<table><tr><td>Method</td><td>Acc.</td><td>Memory</td><td>Tokens</td></tr><tr><td>Ground VQA [3]</td><td>48.7%</td><td>1.29 MB/min</td><td></td></tr><tr><td>ReKV-LLaVAOneVision 0.5B [7]</td><td>50.0%</td><td>68.92 MB/min</td><td>12,544</td></tr><tr><td>ReKV-LLaVAOneVision 7B [7]</td><td>56.0%</td><td>329.64 MB/min</td><td>12,544</td></tr><tr><td>Lando et al. Gemini-based [8]</td><td>56.0%</td><td>3.57 KB/min</td><td>10,880</td></tr><tr><td>PARENT (Ours)</td><td>56.4%</td><td>66.07 KB/min</td><td>3,600</td></tr></table>

Overall, the results show that the most effective strategy is not simply the one with the highest retrieval metrics, but the one that best couples retrieval precision with readable context.

## B. Retrieval Budget and Temporal Coverage

Figure 4 reports the retrieval budget curve for the bestperforming PARENT configuration. Increasing K steadily improves temporal coverage: Hit@K rises from 53.4% to 98.2%, and Recall@K rises from 42.7% to 97.9%. Accuracy, however, follows a different trend. It peaks at 56.4% for K = 12, then decreases as additional captions are included.

This result highlights that temporal recall is useful but not sufficient for answer accuracy. In egocentric video, repeated rooms, objects, screens, tables, and actions create plausible hard negatives. A larger retrieved context can therefore cover more of the annotated evidence while also introducing distractors that make option selection harder for the reasoner.

## C. Comparison with State-of-the-Art Methods

Table II compares our best-performing configuration with recent state-of-the-art methods on QAEgo4D-Closed [3]. PAR-ENT reaches 56.4% accuracy, which is on par with the best previously reported result, 56.0%, obtained by the Geminibased pipeline proposed in [8]. The main advantage of our approach is its substantially lower query-time token usage. While the Gemini-based method of [8] uses the full textual memory, corresponding to 10,880 tokens at query time, our method requires only 3,600 tokens on average. This corresponds to roughly 33% of the full-memory input. Moreover, our memory footprint remains in the KB/min range, with 66.07 KB/min, whereas visual-memory and KV-cache-based methods such as Ground VQA [3] and ReKV [7] operate in the MB/min range. Although these memory measurements are not directly equivalent across methods, since they represent different memory representations, they highlight that our textualmemory approach remains compact while being competitive with substantially heavier visual-memory baselines.

## D. Qualitative Examples

Figure 5 shows representative success and failure cases of our RAG pipeline. In the success case, the retrieved captions explicitly mention the target object and its location, allowing the model to answer that the scissors were placed on the table correctly. In the failure case, the retrieved captions are semantically related to the query, but only provide coarse evidence about the knife being near the sink. As a result, the model predicts “in the sink” instead of the more precise ground-truth location, “in the strainer.” These examples suggest that our method is effective when retrieved captions contain explicit evidence, but suffers from missing or ambiguous fine-grained spatial details.

## V. CONCLUSION

This study presented a training-free textual-memory pipeline for egocentric video question answering. Our best configuration, PARENT, retrieves fine-grained caption subchunks and expands them to their parent captions, achieving 56.4% accuracy on QAEgo4D-Closed while using a substantially smaller query-time context than full-memory reasoning.

Our analysis shows that temporal retrieval coverage does not necessarily translate into better answer accuracy. As the retrieval budget increases, Hit@K and Recall@K continue to improve, but accuracy peaks at an intermediate budget, since additional captions can introduce plausible distractors in long egocentric videos with repeated objects, locations, and actions.

These findings highlight the importance of retrieval design in textual-memory-based egocentric QA. Future work should move beyond treating captions as independent text units and develop evidence-aware episodic memories that preserve temporally grounded events, uncertainty, and answer-relevant support for downstream reasoning.

[1] E. Tulving et al., “Episodic and semantic memory,” Organization of memory, vol. 1, no. 381-403, p. 1, 1972.

[2] Ego4D Consortium, “Ego4d: Around the world in 3,000 hours of egocentric video,” 2022. [Online]. Available: https://arxiv.org/abs/2110.07058

[3] S. Di and W. Xie, “Grounded question-answering in long egocentric videos,” in CVPR, 2024.

[4] J. Shen, J. J. Dudley, and P. O. Kristensson, “Encode-store-retrieve: Augmenting human memory through language-encoded egocentric perception,” in 2024 IEEE International Symposium on Mixed and Augmented Reality (ISMAR). IEEE, 2024, pp. 923–931.

[5] Y. Wang, Y. Yang, and M. Ren, “Lifelongmemory: Leveraging llms for answering queries in long-form egocentric videos,” arXiv preprint, 2024, arXiv:2312.05269. [Online]. Available: https://arxiv.org/abs/2312.05269

[6] J. Yang, S. Liu, H. Guo, Y. Dong, X. Zhang, S. Zhang, P. Wang, Z. Zhou, B. Xie, Z. Wang, B. Ouyang, Z. Lin, M. Cominelli, Z. Cai, Y. Zhang, P. Zhang, F. Hong, J. Widmer, F. Gringoli, L. Yang, B. Li, and Z. Liu, “Egolife: Towards egocentric life assistant,” 2025, arXiv:2503.03803. [Online]. Available: https://arxiv.org/abs/2503.03803

[7] S. Di, Z. Yu, G. Zhang, H. Li, T. Zhong, H. Cheng, B. Li, W. He, F. Shu, and H. Jiang, “Streaming video question-answering with in-context video kv-cache retrieval,” arXiv preprint, 2025, arXiv:2503.00540. [Online]. Available: https://arxiv.org/abs/2503.00540

[8] G. Lando, R. Forte, G. M. Farinella, and A. Furnari, “How far can off-the-shelf multimodal large language models go in online episodic memory question answering?” CoRR, vol. abs/2506.16450, June 2025. [Online]. Available: https://doi.org/10.48550/arXiv.2506.16450

[9] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Kuttler, M. Lewis, W.-t. Yih, T. Rockt ¨ aschel, S. Riedel, and D. Kiela,¨ “Retrieval-augmented generation for knowledge-intensive nlp tasks,” in Advances in Neural Information Processing Systems, vol. 33, 2020, pp. 9459–9474.

[10] L. Barmann and A. Waibel, “Where did i leave my keys? — episodic-¨ memory-based question answering on egocentric videos,” in 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2022, pp. 1559–1567.

[11] H. Ye, H. Zhang, E. Daxberger, L. Chen, Z. Lin, Y. Li, B. Zhang, H. You, D. Xu, Z. Gan, J. Lu, and Y. Yang, “Mm-ego: Towards building egocentric multimodal llms for video qa,” in International Conference on Learning Representations, 2025. [Online]. Available: https://arxiv.org/abs/2410.07177

[12] Z. Ma, C. Gou, H. Shi, B. Sun, S. Li, H. Rezatofighi, and J. Cai, “Drvideo: Document retrieval based long video understanding,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025, pp. 18 936–18 946.

[13] X. Ren, L. Xu, L. Xia, S. Wang, D. Yin, and C. Huang, “Videorag: Retrieval-augmented generation with extreme longcontext videos,” 2025, arXiv:2502.01549. [Online]. Available: https://arxiv.org/abs/2502.01549

[14] Y. Luo, X. Zheng, G. Li, S. Yin, H. Lin, C. Fu, J. Huang, J. Ji, F. Chao, J. Luo, and R. Ji, “Video-rag: Visually-aligned retrieval-augmented long video comprehension,” in Advances in Neural Information Processing Systems, 2025. [Online]. Available: https://arxiv.org/abs/2411.13093

[15] M. Z. U. Alam and H. Zamani, “Open-ended and knowledgeintensive video question answering,” 2025, arXiv:2502.11747. [Online]. Available: https://arxiv.org/abs/2502.11747

[16] Z. Guo, L. Xia, Y. Yu, T. Ao, and C. Huang, “Lightrag: Simple and fast retrieval-augmented generation,” 2024, arXiv:2410.05779. [Online]. Available: https://arxiv.org/abs/2410.05779

[17] B. J. Gutierrez, Y. Shu, Y. Gu, M. Yasunaga, and Y. Su, “Hipporag:´ Neurobiologically inspired long-term memory for large language models,” in Advances in Neural Information Processing Systems, 2024. [Online]. Available: https://arxiv.org/abs/2405.14831

[18] S. Sun, K. Han, Y. Huang, W. Cai, and J. Song, “Egograph: Temporal knowledge graph for egocentric video understanding,” 2026, arXiv:2602.23709. [Online]. Available: https://arxiv.org/abs/2602.23709

[19] K. Kim, Y. Yang, S. Kim, W. Yeo, Y. Lee, M. Ren, and S. J. Hwang, “Ma-egoqa: Question answering over egocentric videos from multiple embodied agents,” 2026, arXiv:2603.09827. [Online]. Available: https://arxiv.org/abs/2603.09827

[20] S. Rajesh, P. Holur, C. Duan, D. Chong, and V. Roychowdhury, “Beyond fact retrieval: Episodic memory for rag with generative semantic workspaces,” 2026, aAAI 2026 Oral; arXiv:2511.07587.

[21] S. Bai, Y. Cai, R. Chen, K. Chen, X. Chen, Z. Cheng, L. Deng, W. Ding, C. Gao, C. Ge, W. Ge, Z. Guo, Q. Huang, J. Huang, F. Huang, B. Hui, S. Jiang, Z. Li, M. Li, M. Li, K. Li, Z. Lin, J. Lin, X. Liu, J. Liu, C. Liu, Y. Liu, D. Liu, S. Liu, D. Lu, R. Luo, C. Lv, R. Men, L. Meng, X. Ren, X. Ren, S. Song, Y. Sun, J. Tang, J. Tu, J. Wan, P. Wang, P. Wang, Q. Wang, Y. Wang, T. Xie, Y. Xu, H. Xu, J. Xu, Z. Yang, M. Yang, J. Yang, A. Yang, B. Yu, F. Zhang, H. Zhang, X. Zhang, B. Zheng, H. Zhong, J. Zhou, F. Zhou, J. Zhou, Y. Zhu, and K. Zhu, “Qwen3-vl technical report,” 2025. [Online]. Available: https://arxiv.org/abs/2511.21631

[22] G. Team, “Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities,” 2025. [Online]. Available: https://arxiv.org/abs/2507.06261

[23] NVIDIA, “Llama 3.2 NeMo Retriever Embedding 300M v1,” NVIDIA NIM model card, 2025, https://docs.api.nvidia.com/nim.

[24] ——, “NVIDIA Retrieval QA Text Reranking Mistral 4B,” NVIDIA NIM model card, https://docs.api.nvidia.com/nim/reference/nvidiarerank-qa-mistral-4b.