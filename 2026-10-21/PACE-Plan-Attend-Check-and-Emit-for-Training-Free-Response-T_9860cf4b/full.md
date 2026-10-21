# PACE: Plan, Attend, Check, and Emit for Training-Free Response Timing in Streaming Video Understanding

Siqi Guo<sup>1</sup>, Yuxia Chen<sup>2</sup>, Wenbo Zhu<sup>3</sup>, Yangguang Ji<sup>4</sup>, Zhenxiang Jiang<sup>5</sup>, Teng Zhong<sup>6</sup>, and Yu Zhang<sup>7</sup>

<sup>1</sup> Nanyang Technological University

<sup>2</sup> University of Electronic Science and Technology of China

<sup>3</sup> University of California, Berkeley

4 Nanyang Technological University

5 National University of Singapore <sup>6</sup> Nanjing Normal University

<sup>7</sup> Beijing Freedo Technology

Abstract. Streaming video understanding requires a model to decide not only what to answer but also when to answer. Evidence arrives progressively, and recent benchmarks identify premature responses, issued before the queried event has concluded, as a recurring failure mode. Existing approaches often learn response timing through additional training, while training-free alternatives typically rely on scalar gating signals that do not explicitly represent when an answer becomes final. We organize chunk-wise streaming errors into three patterns: sampling variation can make an answer unstable, a stable answer may lack support in the observed video, or an answer may be supported by the current evidence while the event remains unfinished. We propose PACE (Plan, Attend, Check, and Emit), a training-free framework designed around these three failure modes. Given a question, PACE drafts a plan specifying what to observe and what will signal completion. As the stream unfolds, it invokes an of-the-shelf video LLM at informative moments, evaluates each draft for stability, grounding in its cited frames, and completion, and records verified observations in an event ledger. Experiments on ProactiveVideoQA and OVO-Bench Forward show that PACE improves over the oficial protocols by up to 5.9 average PAUC points and 6.2 average accuracy points across proprietary and open-source backbones, yielding state-of-the-art performance.

Keywords: Streaming Video Understanding · Video Large Language Models · Proactive Video Question Answering

## 1 Introduction

Video large language models are increasingly being considered for live-streaming settings, including narrating an egocentric camera, monitoring a surveillance feed, and answering questions during playback [4,22,40,12,42]. In these settings, video arrives continuously and interaction takes place while the content is still unfolding. Unlike ofline video question answering, where the full video is available before inference, streaming requires a model to decide not only what to answer but also when to answer and when to remain silent, as relevant evidence may arrive seconds or minutes after a question is asked (Fig. 1).

![](images/f9a83017c30d3d8f797af47d0831a2dd376e09897c3209b207a049bfaf646e82.jpg)  
Fig. 1. Response timing on a video stream: evidence for a question posed at t<sub>0</sub> arrives over time. A chunk-wise baseline answers at every chunk; its first two replies are premature, and it cannot know that its third is final. PACE holds the draft until its watch plan’s completion signal fires, then answers once, correctly.

Most prior work learns response timing through silence-token objectives, scoring heads, or dedicated decision modules [4,30,28,22,40,23], while existing training-free alternatives gate responses using scalar signals such as an embedding delta or a token-drop ratio [15,41]. These approaches have complementary limitations. Learned timing components are dificult to transfer directly to proprietary black-box models, even though those models perform strongly under naive chunk-wise inference on ProactiveVideoQA [29]. Scalar gating signals, meanwhile, do not explicitly represent whether the evidence required by a particular question is complete.

Across recent benchmarks, premature answering emerges as a prominent error. OVO-Bench asks a model to withhold its response until enough future evidence has arrived and finds this ability to be the weakest of the three it evaluates [20]; StreamingBench and OmniMMI report a related ofline-to-streaming gap [17,31]; and 92.7% of Gemini 3 Pro’s incorrect proactive triggers on IP-

IBench occur before the queried event has finished [15]. These results suggest that response timing, rather than visual perception alone, is often the immediate source of error. Self-consistency [27] does not directly address this issue: for the question in Fig. 1, when asked mid-recipe, a model may repeatedly answer “two,” which is consistent with the evidence observed so far but incorrect as a final answer because a third seasoning has yet to appear. Finality therefore depends on both the semantics of the question and the progress of the event; agreement among successive answers alone does not establish it.

We organize chunk-wise streaming errors into three recurring failure modes: decoding noise, stable answers that lack visual support, and supported but premature answers. Based on this taxonomy, we propose PACE (Plan, Attend, Check, and Emit), a training-free framework that coordinates calls to an ofthe-shelf video LLM and applies a corresponding check to each failure mode. Given the question, Plan drafts a watch plan that specifies what to observe and which visual signal will indicate completion; the plan can subsequently be revised under constrained rules. Attend schedules backbone calls at informative moments in the stream. Check evaluates each draft for stability, grounding, and completion: stability voting reduces sensitivity to decoding noise, re-derivation from cited frames targets answers without visual support, and the completion check tests whether a consistent answer is still premature. Emit records replies, supporting evidence, and plan revisions in an event ledger, reducing duplicate outputs and carrying prior evidence forward.

Experiments on ProactiveVideoQA and OVO-Bench Forward show that PACE improves over the oficial protocols by up to 5.9 average PAUC points and 6.2 average accuracy points across proprietary and open-source backbones, yielding state-of-the-art performance.

Our contributions are as follows:

– We organize the premature responses of streaming video LLMs into a threeway taxonomy, and identify consistent prematureness as a stream-specific failure that no inspection of the answer itself can detect.

– We propose PACE, a training-free and model-agnostic framework whose checks match the taxonomy one-to-one, built around a self-revising watch plan and an auditable event ledger.

– PACE achieves state-of-the-art results on ProactiveVideoQA and OVO-Bench Forward across proprietary and open-source backbones.

## 2 Related Work

Video Large Language Models. Modern video LLMs such as the Qwen-VL series and LLaVA-Video combine a vision encoder with a strong language model and are tuned on large video instruction corpora [2,48,1,44]. Their capabilities extend from temporal grounding [33] and user-generated-video repurposing [38] to intelligent image editing [34]. Parallel visual research preserves edge and frequency detail in remote sensing and MRI [5,8,3], or organizes representations through hierarchical semantic alignment [52]. Agentic methods further plan frame selection and tool use over a complete video [26]. Most of these systems assume that the full video is available before generation, while several highperforming models are accessible only through black-box APIs. PACE instead considers frames that arrive causally in a stream.

Streaming Video Understanding. Recent benchmarks report a substantial gap between ofline and streaming video understanding. Models that perform well with complete videos often degrade when frames arrive incrementally, and withholding a response until future evidence becomes available remains particularly challenging [17,31,29,20]. One line of work incorporates streaming behavior through interleaved silence and response, streaming memory, system-level designs, or response-timing heads [4,42,12,43,22,30,28,23,9].

Training-free methods reduce inference cost through KV-cache retrieval and token dropping [7,41], or schedule perception around anticipated evidence [18,39]. More broadly, adaptation has been pursued through feature regularization and label-free prompting [47,53], selective vision-language subspaces [54], cross-modal alignment [56], and test-time prompt updates [49]. Analyses of in-context learning and SFT generalization further motivate flexible inference-time control [35,37]. Methods that explicitly model response release commonly use a decoding-time score [40], a token-drop ratio [41], or similarity between watch proposals and recent frames [15]. These signals do not explicitly encode question-dependent completion conditions. Online action detection estimates event boundaries [10], but not whether the evidence needed for a particular answer is complete.

PACE is also related to test-time verification methods based on repeated sampling, consistency, or visual reinforcement [27,50,19,6]. Evidence-based approaches mitigate hallucinations or abstain when an answer is unreliable [32,46,13], while complementary interventions target safety alignment [51], jailbreak defense [55], or removal of unwanted concepts [36]. These methods operate under a fixed context. In a stream, however, a currently supported answer need not be final [20,15]. PACE therefore adds constrained watch-plan revision, an explicit completion check, and an event ledger for verified observations.

## 3 Method

Sect. 3.1 formalizes the problem and the error taxonomy; Sects. 3.2 to 3.5 then present the four stages of PACE shown in Fig. 2.

## 3.1 Preliminaries

A video stream is a frame sequence $\{ f _ { 1 } , f _ { 2 } , \ldots \}$ revealed causally: at time t the system observes only the prefix $f _ { \leq t }$ . The backbone is a video LLM accessed as a black box, a function M that maps a frame set F and a text prompt p to a text output $M ( F , p )$ , decoded greedily or sampled; PACE interacts with the backbone only through such calls. A question q arrives at time $t _ { 0 }$ , and the system outputs timestamped replies $\{ ( \tau _ { j } , y _ { j } ) \}$ online: at any step it may stay silent or emit a reply, and emitted replies cannot be retracted.

![](images/2dd3bf85ea69d330c249c18f4b3f019118582383a622b46587eeded177ead8d3.jpg)  
Fig. 2. PACE plans what to observe, schedules activity-triggered probes, checks each draft, and records verified events and evidence for subsequent probes.

In our analysis of chunk-wise baselines, incorrect replies follow three recurring patterns: same-timestep resamples disagree because of decoding noise; a stable reply lacks support in the video observed so far; or a stable, supported reply concerns an event that has not yet concluded. These underlying faults can overlap in a single reply. For analysis, we therefore assign labels with a fixed decision tree, disagreement first, followed by support and completion, and give each incorrect reply the first applicable label.

## 3.2 Plan: a Self-Revising Watch Plan

When q arrives, one text-only call ${ \mathcal { M } } ( \emptyset , q )$ produces a watch plan $\pi = ( a , E , c , \rho ) \colon$ answer type $^ { a , }$ evidence checklist E, completion signal $c ,$ and reporting mode $\rho \in \{ \mathrm { r o l l i n g , f i n a l } \}$ . State questions admit a current answer, cumulative questions grow until $c ,$ and reveal questions become answerable only after the relevant evidence appears. Monitor instructions reuse these types and produce one reply per event instance. The question wording determines $\rho \colon$ definite totals and outcomes use final mode, while descriptive requests use rolling mode. The type and mode remain fixed throughout the stream.

PACE permits two plan revisions. Checklist items may be added to $E$ at any probe. The completion signal may be re-bound once, when the probe reports a setting mismatch and the scheduler detects activity in the preceding k chunks. The replacement must be observable by the scheduler. Each revision and its reason are recorded in the ledger; all other plan components remain unchanged.

## 3.3 Attend: Perception-Gated Scheduling

The stream advances in fixed chunks, and a lightweight activity signal $_ { g _ { t } , }$ the mean absolute diference of consecutive frames, determines when to invoke the backbone. Quiet stretches are skipped up to a bounded maximum silence, while increased activity raises the probe density. Two signal sets serve diferent purposes. The probe-trigger set combines pixel and subtitle changes, with subtitles taken from the benchmark’s own track, and uses permissive thresholds to retain probes for dialogue-driven reveals. The corroboration set comprises motion cessation or a scene boundary after sustained activity and uses stricter thresholds to reduce completion triggers from isolated cuts in fast-edited content. For monitor questions, an onset is defined as activity rising above the probe threshold after a quiet stretch. These signals control probe timing and provide corroboration, but emission remains subject to the checks in the next stage.

## 3.4 Check: the Three Checks

At probe time t, PACE makes a greedy call $\mathcal { M } ( F _ { t } , p _ { t } )$ , where $F _ { t }$ contains the current chunk and evidence bank $B _ { t }$ , and $p _ { t }$ contains $q ,$ plan π, and prior replies. The call returns a draft $y _ { t }$ , cited frames $V _ { t } \subseteq F _ { t }$ , checklist progress $\hat { E } _ { t }$ , completion flag $\hat { c } _ { t }$ , setting match, and, for cumulative or monitor questions, newly observed event instances with cited frames. Abstentions and drafts with no new information produce no output. Otherwise, the draft must pass the following three checks during normal operation.

Stability. A draft is new if it difers from both the last emitted reply and the pending draft under a type-aware comparator; the first draft is always new. For a new draft, PACE draws $n - 1$ additional samples at the same timestep and compares all samples with the greedy draft. Counts use numeric equality, multiple-choice answers use option identity, and free text uses token-level F1 above $\theta _ { 2 }$ . The draft passes when at least a fraction $\theta _ { 1 }$ agree. A draft that passes stability but fails a later check becomes pending. If the same pending draft recurs at the next probe, it passes stability without resampling; a draft that previously failed stability is sampled again.

Completion. State questions and rolling reports, including rolling counts, pass by definition. A final reply to a cumulative or reveal question passes if and only if

$$
\left( \hat { c } _ { t } \wedge \mathrm { c o r r } _ { t } \right) \vee \mathrm { v e r } _ { t } ( E ) ,\tag{1}
$$

where $\operatorname { c o r r } _ { t }$ indicates a candidate completion event since the previous probe. The term $\mathrm { v e r } _ { t } ( E )$ holds when every checklist item $\textit { e } \in \textit { E }$ is confirmed by a context-isolated call $\mathcal { M } ( F _ { t } , e )$ that does not receive $y _ { t }$ or $\hat { E } _ { t }$ . For cumulative and reveal questions, E includes a completion item. Monitor replies use the same rule per event instance: an emitted and corroborated instance closes, and the next corroborated onset opens a new one. Completion prompts ask whether the relevant activity has visibly ended, rather than whether the model knows the answer.

Grounding. A context-isolated call $\mathcal { M } ( V _ { t } , q )$ re-derives the answer from the cited frames and question. The draft passes if the result matches $y _ { t }$ under the same comparator with threshold $\theta _ { 2 } ;$ a draft with no cited frames fails. For cumulative questions, grounding is applied to each ledger increment rather than to the final sum. Using the same backbone for drafting and verification may preserve a shared misperception.

End-of-stream fallback. If the stream ends before any reply is emitted, PACE releases the pending draft with the highest vote share. If no draft passed stability, it releases the last greedy draft. This fallback can therefore release a draft that has not passed all three checks.

## 3.5 Emit: the Event Ledger

Every emitted reply $( \tau _ { j } , y _ { j } )$ , event observation, and plan revision enters the ledger at the end of its probe. The ledger has three functions. First, it reduces duplication: drafts suficiently similar to emitted content are suppressed, and an observation that overlaps an existing entry in time and cited frames is merged rather than added. Second, it supports counting: each observation that passes the stability and grounding checks becomes an entry whether or not a reply is emitted, and cumulative totals are read from the ledger rather than re-counted from pixels at every probe, a known dificulty in streaming counting [20]. Third, it retains evidence for subsequent probes and audit: cited frames from replies and entries form the evidence bank $B _ { t }$ , which uses a fixed capacity with oldest-first eviction, and each plan revision is stored with its reason.

## 4 Experiments

## 4.1 Setup

Benchmarks and protocol. We evaluate on ProactiveVideoQA [29] and OVO-Bench Forward Active Responding [20]. ProactiveVideoQA contains 1,377 videos and 3,510 reply turns from web videos, egocentric videos, TV series, and anomaly detection videos. We use its oficial PAUC metric with ω=0.5 and the oficial GPT-4.1 judge. OVO-Bench evaluates repetition counting, step recognition, and clues-reveal responding with per-probe accuracy. We follow the oficial temporal settings for both benchmarks. ProactiveVideoQA uses 2 s chunks at $2 \mathrm { f p s }$ for web videos and 5 s chunks at 1 fps for the other domains. OVO-Bench uses fixed probe timestamps. At each OVO-Bench probe, PACE reports the rolling ledger count for repetition counting, the current state for step recognition, and yes for clues-reveal responding when all checks pass. Only the clues-reveal task uses the completion check, while model-drafted question types are evaluated only on ProactiveVideoQA.

Table 1. PAUC on ProactiveVideoQA [29] with ω=0.5. Published results are taken from the corresponding papers. The bottom block compares each backbone with and without PACE.
<table><tr><td>Model or method</td><td>WEB</td><td>EGO</td><td>TV</td><td>VAD</td><td>Avg.</td></tr><tr><td>Human [29]</td><td>38.6</td><td>38.2</td><td>47.0</td><td>53.6</td><td>44.4</td></tr><tr><td>LLaVA-OneVision-7B [14] 55.0</td><td></td><td>61.6</td><td>45.1</td><td>25.6</td><td>46.8</td></tr><tr><td>VideoLLM-online [4]</td><td>25.9</td><td>25.0</td><td>18.3</td><td>25.0</td><td>23.6</td></tr><tr><td>MMDuet [30]</td><td>38.9</td><td>46.0</td><td>21.1</td><td>27.4</td><td>33.4</td></tr><tr><td>MMDuet2-RL [28]</td><td>53.3</td><td>33.6</td><td>43.4</td><td>28.9</td><td>39.8</td></tr><tr><td>Em-Garde [45]</td><td>44.3</td><td>52.3</td><td></td><td>27.4</td><td></td></tr><tr><td>VideoChat3-4B [16]</td><td>28.4</td><td>28.1</td><td>34.7</td><td>25.1</td><td>29.1</td></tr><tr><td>MOSS-VL-Realtime e [25]</td><td>55.4</td><td>47.8</td><td>50.2</td><td>35.3</td><td>47.2</td></tr><tr><td colspan="6">PACE with Different Backbones</td></tr><tr><td>GPT-4.1 [29]</td><td>51.7</td><td>58.8</td><td>56.8</td><td>46.2</td><td>53.4</td></tr><tr><td>+ PACE</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-4.1-mini [29]</td><td> ${ \begin{array} { l l l } { 5 9 . 2 \left( + 7 . 5 \right) } & { 6 4 . 7 \left( + 5 . 9 \right) } & { 6 2 . 1 \left( + 5 . 3 \right) } & { 5 1 . 1 \left( + 4 . 9 \right) } & { 5 9 . 3 \left( + 5 . 9 \right) } \\ { 4 7 . 8 } & { 6 5 . 8 } & { 5 9 . 4 } & { 4 7 . 7 } & { 5 5 . 2 } \end{array} }$ </td><td></td><td></td><td></td><td></td></tr><tr><td>+ PACE</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { Q w e n 2 . 5 - V L - 7 B \ [ 2 ] }$ </td><td></td><td></td><td></td><td></td><td> $\begin{array} { l } { { 5 4 . 1 _ { ( + 6 . 3 ) } \ 7 0 . 2 _ { ( + 4 . 4 ) } \ 6 2 . 8 _ { ( + 3 . 4 ) } \ 5 6 . 9 _ { ( + 9 . 2 ) } \ 6 1 . 0 _ { ( + 5 . 8 ) } } } \\ { { 5 2 . 7 \qquad 4 6 . 5 \qquad 3 6 . 5 \qquad 2 9 . 3 \qquad 4 1 . 3 } } \end{array}$ </td></tr><tr><td> $\mathbf { \Sigma } + \mathbf { \Sigma } \mathbf { P } \mathbf { A } \mathbf { C } \mathbf { E }$ </td><td></td><td></td><td></td><td></td><td> $5 9 . 9 _ { ( + 7 . 2 ) } 5 4 . 8 _ { ( + 8 . 3 ) } 4 0 . 1 _ { ( + 3 . 6 ) } 3 3 . 7 _ { ( + 4 . 4 ) } 4 7 . 1 _ { ( + 5 . 8 ) }$ </td></tr></table>

Models, baselines, and implementation. On ProactiveVideoQA, we evaluate PACE with Qwen2.5-VL-7B [2], GPT-4.1, and GPT-4.1-mini. On OVO-Bench, we use Qwen2-VL-7B [24], GPT-4o [21], and Gemini 1.5 Pro [11]. On each benchmark the backbones are chosen among models with oficially reported baselines, so every unwrapped number has a published reference. The default setting uses $n { = } 3 , \theta _ { 1 } { = } 2 / 3 , \theta _ { 2 } { = } 0 . 6 .$ , a re-binding window of k=3 chunks, vote temperature 0.7, and greedy temperature 0. Maximum probe silence is 20 s and is reduced to 10 s for anomaly detection videos. Other thresholds, prompts, frame budgets, and evidence-bank limits are provided with the code. We run stochastic configurations with three seeds and report the mean. The main tables compare PACE with published systems and pair each PACE result with its unmodified backbone; all matched comparisons use the same chunking and frame budget.

## 4.2 Main Results

PACE lifts every backbone on ProactiveVideoQA. Table 1 shows consistent gains: GPT-4.1 improves from 53.4 to 59.3, GPT-4.1-mini from 55.2 to 61.0, and Qwen2.5-VL-7B from 41.3 to 47.1. The average gain is nearly constant, 5.8 to 5.9 points, although the three backbones difer in scale, family, and access, and the weakest of them benefits as much as the strongest. The wrapped GPT-4.1-mini sets the best average reported on the benchmark, 5.8 points above the previous best, its own backbone under the oficial chunk-wise protocol, and 13.8 points above the strongest purpose-built streaming system. The human reference row trails several systems, an artifact of live annotation latency under a metric that penalizes delayed replies.

Table 2. Accuracy on OVO-Bench Forward Active Responding [20]. Published results are taken from the corresponding papers. The bottom block compares each backbone with and without PACE.
<table><tr><td>Model or method</td><td>REC</td><td>SSR</td><td>CRR</td><td>Avg.</td></tr><tr><td>Human [20]</td><td>95.5</td><td>89.7</td><td>93.6</td><td>92.9</td></tr><tr><td>LLaVA-Video-7B [44]</td><td>34.1</td><td>70.0</td><td>60.4</td><td>54.8</td></tr><tr><td>Qwen2-VL-72B [24]</td><td>38.8</td><td>64.1</td><td>45.0</td><td>49.3</td></tr><tr><td>Flash-VStream-7B [42]</td><td>8.0</td><td>67.3</td><td>60.0</td><td>45.1</td></tr><tr><td>Dispider-7B [22]</td><td>18.1</td><td>37.4</td><td>48.8</td><td>34.7</td></tr><tr><td>Em-Garde [45]</td><td></td><td></td><td></td><td>31.0</td></tr><tr><td>MOSS-VL-Realtime [25]</td><td></td><td></td><td></td><td>62.1</td></tr><tr><td colspan="5">PACE with Different Backbones</td></tr><tr><td>Gemini 1.5 Pro [11]</td><td>35.5</td><td>74.2</td><td>61.7</td><td>57.2</td></tr><tr><td>+ PACE</td><td></td><td></td><td></td><td> $4 5 . 1 _ { ( + 9 . 6 ) } 7 9 . 9 _ { ( + 5 . 7 ) } 6 5 . 2 _ { ( + 3 . 5 ) } 6 3 . 4 _ { ( + 6 . 2 ) }$ </td></tr><tr><td>GPT-4o [21]</td><td>27.6</td><td>73.2</td><td>59.4</td><td>53.4</td></tr><tr><td>+ PACE</td><td></td><td></td><td></td><td> $3 5 . 2 ( + 7 . 6 ) \ 7 4 . 1 ( + 0 . 9 ) \ 6 3 . 8 ( + 4 . 4 ) \ 5 7 . 7 ( + 4 . 3 )$ </td></tr><tr><td> $\mathrm { Q w e n 2 - V L - 7 B \ [ 2 4 ] }$ </td><td>31.7</td><td>65.8</td><td>48.8</td><td>48.7</td></tr><tr><td> $\mathbf { \Sigma } + \mathbf { \Sigma } \mathbf { P } \mathbf { A } \mathbf { C } \mathbf { E }$ </td><td></td><td></td><td></td><td> $3 3 . 9 _ { ( + 2 . 2 ) } 7 0 . 5 _ { ( + 4 . 7 ) } 5 0 . 2 _ { ( + 1 . 4 ) } 5 1 . 5 _ { ( + 2 . 8 ) }$ </td></tr></table>

Where the gains land difers by backbone. For GPT-4.1 and Qwen2.5- VL-7B the largest improvements come on WEB and EGO, up to 7.5 and 8.3 points, where frequent activity keeps rolling answers current and the ledger suppresses repetition over long horizons. GPT-4.1-mini instead gains most on VAD, 9.2 points: anomaly clips reward a system that stays silent until the anomaly has run its course, and the completion machinery converts that patience into score. TV shows the smallest gains for GPT-4.1-mini and Qwen2.5-VL-7B, 3.4 and 3.6 points, and is among the smallest for GPT-4.1, suggesting that dialogue-driven reveals remain the hardest timing problem.

OVO-Bench Forward shows the same picture. In Table 2, all three backbones improve, Gemini 1.5 Pro from 57.2 to 63.4, GPT-4o from 53.4 to 57.7, and Qwen2-VL-7B from 48.7 to 51.5, and the wrapped Gemini passes the strongest published system, MOSS-VL-Realtime at 62.1. For the two proprietary backbones the largest gains fall on repetition counting, 9.6 points for Gemini and 7.6 for GPT-4o, the subtask that probes cross-time memory and the one the ledger is built for. The weaker Qwen2-VL-7B instead gains most on step recognition, 4.7 points, where the stability vote filters its noisier drafts. Cluesreveal responding improves on every backbone, by 1.4 to 4.4 points. The gap to human performance remains wide, 63.4 against 92.9, and most of it sits on repetition counting.

Summary. Three facts hold across both benchmarks. PACE improves all six backbones, by 2.8 to 6.2 average points, under one configuration shared across backbones and benchmarks, apart from the domain-level silence bound stated in Sect. 4.1. The wrapped systems set the best reported averages on both benchmarks. And the gains concentrate where each mechanism is designed to pay: counting subtasks for the ledger, event-bounded domains for the completion machinery, and noisy drafts for the stability vote. The strongest baseline also remains the strongest wrapped system, so PACE adds response timing on top of perception quality rather than trading one for the other.

## 4.3 Visualization

Fig. 3 traces PACE and the direct chunk-wise backbone on two representative videos. On the rolling web video both systems follow the scene as it evolves, and their replies coincide while new content keeps arriving. They part ways once the scene stops changing: at 14 s the direct protocol produces another near-duplicate of its previous reply, whereas PACE finds that the draft adds nothing beyond its emitted replies and stays silent. On the anomaly clip the divergence is sharper. The direct backbone describes the explosion at 76 s, while it is still unfolding, so its reply names sparks but misses the smoke that defines the full event. PACE holds the same draft because no completion signal has fired, and emits one complete reply at 80 to 82 s, after the sparks and smoke have run their course. The two cases exhibit the two behaviors the watch plan distinguishes: rolling updates that track the stream, and a final answer released only once the event has concluded.

## 4.4 Ablations

Tables 3 and 4 remove one module at a time: the watch plan, each of the three checks, the plan revision, and the ledger. Ablations are run on ProactiveVideoQA with one closed-source and one open-source backbone, GPT-4.1-mini and Qwen2.5-VL-7B. Because all checks must pass before emission, multiple checks may reject the same compound error, so leave-one-out deltas are a conservative estimate of each module’s contribution. We interpret the deltas relative to the pattern predicted by the taxonomy rather than testing individual cells for significance.

All modules contribute. Removing any single module lowers the average PAUC of both backbones, so the overall gain rests on complementary mechanisms rather than a single device. The two largest drops come from the watch plan, 3.8 points on GPT-4.1-mini and 4.1 on Qwen2.5-VL-7B, and the completion check, 3.9 and 3.6 points. Even without the watch plan, the remaining checks keep the system above its unmodified backbone, 57.2 against 55.2 and 43.0 against 41.3, so the plan is necessary for the full gain but not for all of it.

The completion check matters most where events must end. Its removal costs GPT-4.1-mini 4.8 points on EGO and 5.3 on VAD but only 1.8 on WEB, and Qwen2.5-VL-7B follows the same pattern with 4.2, 4.8, and 2.2 points. WEB questions often admit rolling answers, where a partial reply is already valid output; EGO and VAD hinge on event boundaries, where a model may describe the ongoing action correctly yet cannot fix the final count or outcome before

![](images/e688909a9912598279824526c14fa7501676064341d1ff97a5a7ccb731fd3f47.jpg)  
1.5 s

![](images/59c2669e24c8bb3c2d7f7d24b41aa2b833f26d90d7aa519828e27f0befad3e1b.jpg)  
5.5 s

![](images/4daa04c551a9aa34ef601ea7a799720097c689f46ee0f86383ccb90af4b41aa7.jpg)  
10.0 s

![](images/5884487aaf75cb76e7d76070776a6591395a5b9502c863f7af899c7750c0463a.jpg)  
14.0 s  
Question What are the people doing in the car?

GT 0.0–3.7 s: The man is driving a black convertible car. 4.6–7.4 s: The boy is sitting in the back seat, looking at something and eating something, while the man is driving. 8.8–15.7 s: The man, boy, and girl are all sitting in the car and eating something.

Direct 2 s: driving a black convertible; 6 s: the boy is sitting and eating; 10 s: the man, boy, and girl are eating; 14 s: repeated similar reply

PACE 2 s: emit: driving a black convertible; 6 s: emit: the boy is sitting and eating; 10 s: emit: all three are sitting and eating; 14 s: silent: no new information

![](images/8441eb8a32152b194b9b17d2137bd99883f9fecdf876d7d9bc151f7e89fcc903.jpg)  
20 s

![](images/da881e759081dc065ac2c6a86b93a596ee7d3eee3fe3b2cd7a9969cc922e928d.jpg)  
60 s

![](images/035471c09ef938e939bc2e0133ed666ee0666d96b2ae1d345e96292ef1d672e9.jpg)  
76 s

![](images/6ca19c834848ac157bc7ded01120577304f566ae023394a00b759485ca285fa2.jpg)  
80 s  
Question What suspicious or harmful activities, including unlawful, criminal behaviors or destructive accidents, are happening in the video?  
GT 74.3–81.7 s: The machine exploded, emitting a large amount of sparks and smoke.  
Direct 76 s: the machine is exploding and producing sparks; 80 s: the machine exploded, producing sparks and smoke  
PACE 76 s: hold: the explosion is still unfolding; 80–82 s: emit: the machine exploded, emitting a large amount of sparks and smoke

Fig. 3. Qualitative cases from ProactiveVideoQA: a rolling web video, top, and a singletruth anomaly clip, bottom. Each case shows sampled frames with their timestamps, the ground-truth replies with their reply windows, the direct chunk-wise replies of the backbone, and the behavior of PACE.  
Table 3. Module ablation of PACE on ProactiveVideoQA with the closed-source backbone GPT-4.1-mini. Each row removes one module from the full system; subscripts give the change from the full system.
<table><tr><td>Variant</td><td>WEB</td><td>EGO</td><td>TV</td><td>VAD</td><td>Avg.</td></tr><tr><td>PACE full</td><td>54.1</td><td>70.2</td><td>62.8</td><td>56.9</td><td>61.0</td></tr><tr><td>w/o watch plan</td><td></td><td></td><td></td><td>51.4(−2.7) 66.1 (−4.1) 59.3(−3.5) 51.9 (−5.0) 57.2 (−3.8)</td><td></td></tr><tr><td>w/o stability vote</td><td>52.6 (−1.5) 69.0 (−1.2) 63.0 (+0.2) 55.1 (−1.8) 59.9 (−1.1)</td><td></td><td></td><td></td><td></td></tr><tr><td>w/0 completion check 52.3 (−1.8) 65.4 (−4.8) 59.1 (−3.7) 51.6 (−5.3) 57.1 (−3.9)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/0 grounding check 53.1 (−1.0) 68.5 (−1.7) 61.3 (−1.5) 54.7 (−2.2) 59.4 (−1.6)</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>w/o plan revision</td><td>53.6 (−0.5) 69.1 (−1.1) 62.0 (−0.8) 55.3 (−1.6) 60.0 (−1.0)</td><td></td><td></td><td></td><td></td></tr><tr><td>w/o ledger</td><td></td><td></td><td></td><td>51.7 (−2.4) 68.2 (−2.0) 61.5 (−1.3) 54.8 (−2.1) 59.1 (−1.9)</td><td></td></tr></table>

the action ends. This asymmetry supports the central distinction of Sect. 3.1:

Table 4. Module ablation of PACE on ProactiveVideoQA with the open-source backbone Qwen2.5-VL-7B. Each row removes one module from the full system; subscripts give the change from the full system.
<table><tr><td>Variant</td><td>WEB</td><td>EGO</td><td>TV</td><td>VAD</td><td> $\operatorname { A v g } .$ </td></tr><tr><td>PACE full</td><td>59.9</td><td>54.8</td><td>40.1</td><td>33.7</td><td>47.1</td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  watch plan</td><td>56.4(−3.5)</td><td>50.1 (−4.7)</td><td></td><td>36.3(−3.8) 29.2(−4.5)</td><td> $4 3 . 0 _ { ( - 4 . 1 ) }$ </td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  stability vote</td><td>57.8 (−2.1)</td><td></td><td></td><td>52.8 (−2.0) 39.2 (−0.9) 31.4 (−2.3)</td><td> $4 5 . 3 _ { \left( - 1 . 8 \right) }$ </td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  completion check 57.7 (−2.2)</td><td></td><td>50.6 (−4.2)</td><td></td><td>36.7(−3.4) 28.9 (−4.8)</td><td> $4 3 . 5 _ { \left( - 3 . 6 \right) }$ </td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  grounding check 57.6 (−2.3)</td><td></td><td></td><td></td><td>52.2 (−2.6) 37.7 (−2.4) 30.8 (−2.9)</td><td> $4 4 . 6 _ { \ : ( - 2 . 5 ) }$ </td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  plan revision</td><td>59.1 (−0.8)</td><td>53.4 (−1.4)</td><td>38.9 (−1.2)</td><td> $3 2 . 0 _ { ( - 1 . 7 ) }$ </td><td> $4 5 . 9 _ { ( - 1 . 2 ) }$ </td></tr><tr><td> $\mathrm { w } / \mathrm { o }$  ledger</td><td> $5 6 . 9 _ { \left( - 3 . 0 \right) }$ </td><td>52.4 (−2.4)</td><td> $3 8 . 4 _ { ( - 1 . 7 ) }$ </td><td> $3 1 . 4 _ { ( - 2 . 3 ) }$ </td><td> $4 4 . 8 _ { \left( - 2 . 3 \right) }$ </td></tr></table>

support by the current prefix does not make an answer final, and neither stability nor grounding substitutes for an explicit completion judgment.

Grounding and stability follow the taxonomy. Grounding contributes 1.6 points on GPT-4.1-mini and 2.5 on Qwen2.5-VL-7B, with the largest perdomain loss on VAD. Stability contributes 1.1 and 1.8 points; its removal improves GPT-4.1-mini on TV by only 0.2 points.

The ledger and plan revision extend the gains over time. Removing the ledger costs most on WEB, 2.4 and 3.0 points, consistent with its role in cumulative counting and deduplication. Plan revision contributes 1.0 and 1.2 points, mainly on VAD and EGO, supporting its role in correcting initial-plan mismatches.

Weaker backbones lean on verification, stronger ones on timing. Qwen2.5-VL-7B loses more from every removal except the completion check. This suggests that verification and memory benefit the weaker backbone more, while completion remains important even for the stronger model.

## 5 Conclusion

PACE combines training-free stability, grounding, and completion checks with an auditable event ledger, improving multiple backbones on two proactive benchmarks. Future work should reduce verification cost, diversify verifiers, and learn adaptive watch plans

## Acknowledgments

This work was supported in part by the Chongqing Natural Science Foundation (CSTB2024NSCQ-MSX1026), in part by the Fundamental Research Funds for the Central Universities (No. SWU-KT\_XJ2025002001), in part by the National Key R&D Program of China (No. 2024YFC3808901), and in part by the NSFC (62102328).

## References

1. Bai, S., Cai, Y., Chen, R., et al.: Qwen3-VL Technical Report (2025)

2. Bai, S., Chen, K., Liu, X., et al.: Qwen2.5-VL Technical Report (2025), arXiv:2502.13923

3. Chen, H., Fang, P., Chen, Y., Ren, Y., Hao, J., Tang, F., Cai, X., Shan, S., Liu, F.: Hifi-mamba: Dual-stream?-laplacian enhanced mamba for high-fidelity mri reconstruction. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 2868–2876 (2026)

4. Chen, J., Lv, Z., Wu, S., et al.: VideoLLM-online: Online Video Large Language Model for Streaming Video. In: CVPR. pp. 18407–18418. IEEE (2024)

5. Chen, Y., Fang, P., Zhong, X., Yu, J., Zhang, X., Li, T.: Hi-resnet: Edge detail enhancement for high-resolution remote sensing segmentation. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 17, 15024– 15040 (2024)

6. Dhuliawala, S., Komeili, M., Xu, J., et al.: Chain-of-Verification Reduces Hallucination in Large Language Models. In: Findings of the Association for Computational Linguistics: ACL 2024. pp. 3563–3578. Association for Computational Linguistics (2024)

7. Di, S., Yu, Z., Zhang, G., et al.: Streaming Video Question-Answering with Incontext Video KV-Cache Retrieval. In: ICLR. OpenReview.net (2025)

8. Fang, P., Chen, H., Chen, Y., Sun, T., Liu, J., Cai, X.: Chasm: Cross-frequency harmonized axis-separable mixing for spectral token operators. arXiv preprint arXiv:2605.14727 (2026)

9. Fu, S., Yang, Q., Li, Y., et al.: ViSpeak: Visual Instruction Feedback in Streaming Videos. In: ICCV. pp. 21778–21788. IEEE (2025)

10. Geest, R.D., Gavves, E., Ghodrati, A., Li, Z., Snoek, C., Tuytelaars, T.: Online Action Detection. In: ECCV. pp. 269–284. Springer (2016)

11. Gemini Team, Georgiev, P., Lei, V.I., et al.: Gemini 1.5: Unlocking Multimodal Understanding across Millions of Tokens of Context (2024)

12. Huang, Z., Li, X., Li, J., et al.: Online Video Understanding: OVBench and VideoChat-Online. In: CVPR. pp. 3328–3338. Computer Vision Foundation / IEEE (2025)

13. Kamath, A., Jia, R., Liang, P.: Selective Question Answering under Domain Shift. In: ACL. pp. 5684–5696. Association for Computational Linguistics (2020)

14. Li, B., Zhang, Y., Guo, D., et al.: LLaVA-OneVision: Easy Visual Task Transfer (2024)

15. Li, J., Chen, Y., Song, W., et al.: IPIBench: Evaluating Interactive Proactive Intelligence of MLLMs under Continuous Streams (2026), arXiv:2605.27074

16. Li, X., Zhu, Y., Zeng, X., et al.: VideoChat3: Fully Open Video MLLM for Eficient and Generalist Video Understanding (2026)

17. Lin, J., Fang, Z., Chen, C., et al.: StreamingBench: Assessing the Gap for MLLMs to Achieve Streaming Video Understanding. In: ICASSP. pp. 12147–12151. IEEE (2026)

18. Long, L., He, Y., Ye, W., et al.: Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory (2025), arXiv:2508.09736

19. Manakul, P., Liusie, A., Gales, M.J.F.: SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models. In: EMNLP. pp. 9004–9017. Association for Computational Linguistics (2023)

20. Niu, J., Li, Y., Miao, Z., et al.: OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding? In: CVPR. pp. 18902–18913. Computer Vision Foundation / IEEE (2025)

21. OpenAI: GPT-4o System Card (2024)

22. Qian, R., Ding, S., Dong, X., et al.: Dispider: Enabling Video LLMs with Active Real-Time Interaction via Disentangled Perception, Decision, and Reaction. In: CVPR. pp. 24045–24055. Computer Vision Foundation / IEEE (2025)

23. Wang, H., Feng, B., Lai, Z., et al.: StreamBridge: Turning Your Ofline Video Large Language Model into a Proactive Streaming Assistant. In: NeurIPS. pp. 146647–146674 (2025)

24. Wang, P., Bai, S., Tan, S., et al.: Qwen2-VL: Enhancing Vision-Language Model’s Perception of the World at Any Resolution (2024)

25. Wang, P., Tan, C., Zhou, S., et al.: MOSS-VL Technical Report (2026)

26. Wang, X., Zhang, Y., Zohar, O., Yeung-Levy, S.: VideoAgent: Long-form Video Understanding with Large Language Model as Agent. In: ECCV. pp. 58–76. Springer (2024)

27. Wang, X., Wei, J., Schuurmans, D., et al.: Self-Consistency Improves Chain of Thought Reasoning in Language Models. In: ICLR. OpenReview.net (2023)

28. Wang, Y., Liu, S., Wang, D., et al.: MMDuet2: Enhancing Proactive Interaction of Video MLLMs with Multi-Turn Reinforcement Learning (2025), arXiv:2512.06810

29. Wang, Y., Meng, X., Wang, Y., Zhang, H., Zhao, D.: ProactiveVideoQA: A Comprehensive Benchmark Evaluating Proactive Interactions in Video Large Language Models (2025), arXiv:2507.09313

30. Wang, Y., Meng, X., Wang, Y., et al.: VideoLLM Knows When to Speak: Enhancing Time-Sensitive Video Comprehension with Video-Text Duet Interaction Format. In: EMNLP (Findings). pp. 6338–6359. Association for Computational Linguistics (2025)

31. Wang, Y., Wang, Y., Chen, B., Wu, T., Zhao, D., Zheng, Z.: OmniMMI: A Comprehensive Multi-modal Interaction Benchmark in Streaming Video Contexts. In: CVPR. pp. 18925–18935. Computer Vision Foundation / IEEE (2025)

32. Whitehead, S., Petryk, S., Shakib, V., et al.: Reliable Visual Question Answering: Abstain Rather Than Answer Incorrectly. In: ECCV. pp. 148–166. Springer (2022)

33. Wu, Y., Hu, X., Sun, Y., Zhou, Y., Zhu, W., Rao, F., Schiele, B., Yang, X.: Number it: Temporal grounding videos like flipping manga. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 13754–13765 (2025)

34. Wu, Y., Li, Z., Hu, X., Ye, X., Zeng, X., Yu, G., Zhu, W., Schiele, B., Yang, M.H., Yang, X.: Kris-bench: Benchmarking next-level intelligent image editing models. Advances in Neural Information Processing Systems 38 (2026)

35. Wu, Y., Yang, X.: A glance at in-context learning. Frontiers of Computer Science 18(5), 185347 (2024)

36. Wu, Y., Zhou, S., Yang, M., Wang, L., Chang, H., Zhu, W., Hu, X., Zhou, X., Yang, X.: Unlearning concepts in difusion model via concept domain correction and concept preserving gradient. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 8496–8504 (2025)

37. Wu, Y., Zhou, Y., Zhou, Z., Peng, Y., Ye, X., Hu, X., Zhu, W., Qi, L., Yang, M.H., et al.: On the generalization of sft: A reinforcement learning perspective with reward rectification. In: International Conference on Learning Representations. vol. 2026, pp. 27550–27571 (2026)

38. Wu, Y., Zhu, W., Cao, J., Lu, Y., Li, B., Chi, W., Qiu, Z., Su, L., Zheng, H., Wu, J., et al.: Video repurposing from user generated content: A large-scale dataset

and benchmark. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 8487–8495 (2025)

39. Yang, H., Tang, F., Zhao, L., et al.: StreamAgent: Towards Anticipatory Agents for Streaming Video Understanding (2025), arXiv:2508.01875

40. Yang, Z., Zhang, K., Hu, Y., et al.: LiveStar: Live Streaming Assistant for Real-World Online Video Understanding. In: NeurIPS. pp. 35334–35372 (2025)

41. Yao, L., Li, Y., Wei, Y., et al.: TimeChat-Online: 80% Visual Tokens are Naturally Redundant in Streaming Videos. In: ACM Multimedia. pp. 10807–10816. ACM (2025)

42. Zhang, H., Wang, Y., Tang, Y., Liu, Y., Feng, J., Jin, X.: Flash-Vstream: Eficient Real-Time Understanding for Long Video Streams. In: ICCV. pp. 21059–21069. IEEE (2025)

43. Zhang, P., Dong, X., Cao, Y., et al.: InternLM-XComposer2.5-OmniLive: A Comprehensive Multimodal System for Long-term Streaming Video and Audio Interactions (2024), arXiv:2412.09596

44. Zhang, Y., Wu, J., Li, W., Li, B., Ma, Z., Liu, Z., Li, C.: Llava-video: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713 (2024)

45. Zheng, Y., Ding, X., Yang, Y., et al.: Em-Garde: A Propose-Match Framework for Proactive Streaming Video Understanding (2026)

46. Zhu, X., Fang, J., Wang, S., Zhu, B., Wang, Z., Yang, Y., He, X.: Mitigating hallucinations in large vision-language models without performance degradation. In: Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 1995–2009 (2026)

47. Zhu, X., Wang, S., Lu, J., Hao, Y., Liu, H., He, X.: Boosting few-shot learning via attentive feature regularization. Proceedings of the AAAI Conference on Artificial Intelligence 38(7), 7793–7801 (2024)

48. Zhu, X., Wang, S., Zhu, B., Li, M., Li, Y., Fang, J., Wang, Z., Wang, D., Zhang, H.: Dynamic multimodal prototype learning in vision-language models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 2501–2511 (2025)

49. Zhu, X., Wu, H., Wang, S., Zhu, B., Ge, J., Zhang, J., Chen, L.: Robustifying vision-language models via test-time prompt adaptation. In: Proceedings of the 43rd International Conference on Machine Learning. vol. 306 (2026)

50. Zhu, X., Zhao, K., Yi, L., Wang, S., Wang, Z., Zhu, B., Zhang, H., He, X.: Look carefully: Adaptive visual reinforcements in multimodal large language models for hallucination mitigation. In: International Conference on Learning Representations. vol. 2026, pp. 81235–81256 (2026)

51. Zhu, X., Zhu, B., Fang, J., Wang, S., Zhang, Y., Wang, X., He, X.: GuardAlign: Test-time safety alignment in multimodal large language models. In: International Conference on Learning Representations. vol. 2026, pp. 118865–118890 (2026)

52. Zhu, X., Zhu, B., Li, Y., Fang, J., Wang, S., Zhao, K., Zhang, H.: Hierarchical semantic alignment for image clustering. Proceedings of the AAAI Conference on Artificial Intelligence 40(34), 29177–29185 (2026)

53. Zhu, X., Zhu, B., Tan, Y., Wang, S., Hao, Y., Zhang, H.: Enhancing zero-shot vision models by label-free prompt distribution learning and bias correcting. In: Advances in Neural Information Processing Systems. vol. 37, pp. 2001–2025 (2024)

54. Zhu, X., Zhu, B., Tan, Y., Wang, S., Hao, Y., Zhang, H.: Selective vision-language subspace projection for few-shot CLIP. In: Proceedings of the 32nd ACM International Conference on Multimedia. pp. 3848–3857 (2024)

55. Zhu, X., Zhu, B., Wang, S., Fang, J., Zhao, K., Zhang, H., He, X.: Principled steering via null-space projection for jailbreak defense in vision-language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22889–22899 (2026)

56. Zhu, X., Zhu, B., Wang, S., Zhao, K., Zhang, H.: Enhancing CLIP robustness via cross-modality alignment. In: Advances in Neural Information Processing Systems. vol. 38, pp. 17553–17576 (2025)