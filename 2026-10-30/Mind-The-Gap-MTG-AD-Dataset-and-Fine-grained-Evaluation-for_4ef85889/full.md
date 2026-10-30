# Mind The Gap (MTG-AD): Dataset and Fine-grained Evaluation for End-to-End Inline Audio Descriptions

Subhashini Venugopalan<sup>1</sup>, Jenny Tan<sup>1</sup>, Taylor Roper<sup>1</sup>, Jimmy Tobin<sup>1</sup>, Anton Kast<sup>1</sup>, Alicia Martin<sup>1</sup>, Sam Sepah<sup>1</sup>, Amy Pavel<sup>1,2</sup> <sup>1</sup>Google Research, <sup>2</sup>UC Berkeley

## Abstract

Audio descriptions (AD) are essential for making visual media accessible to blind and low-vision (BLV) users. While Multimodal Large Language Models (MLLMs) offer a scalable solution for generating audio descriptions on-demand, their performance on “inline” audio descriptions – which must fit within existing silences in a video – remains under-explored, particularly for diverse user-generated content (UGC). We present a comprehensive investigation into MLLM-generated inline audio descriptions. First, we introduce MTG-AD, a professionally annotated dataset of 36 hours of videos, each averaging ∼5 mins. and 8k+ descriptions in total, co-created by paired blind and sighted AD professionals to provide a highquality benchmark for this task. Second, we propose MTG-AD-Eval, an evaluation framework based on audio description expert guidelines that provides fine-grained, reference-free actionable metrics for model improvement. Crucially, our framework evaluates the end-to-end task: it renders the generated scripts into audio tracks to assess their fit within the original video’s natural silences, minimizing disruptive overlaps. Our analysis across frontier MLLMs reveals that while models accurately describe visual events and substantially reduce speech overlaps, gaps persist in narrative pacing and flow. Finally, we empirically validate that our automated and reference-free metrics correlate significantly with human quality ratings from sighted raters and 7 BLV participants, finding that while users narrowly prefer human-authored audio descriptions, they find MLLM-generated audio descriptions highly beneficial for comprehension. We provide actionable system design guidance and release MTG-AD at hf.co/datasets/google/mind-the-gap-inline-ad and MTG-AD-Eval at github.com/google/mtg-ad.

## 1 Introduction

Blind and low-vision (BLV) viewers turn to user-generated videos to learn new skills, pursue hobbies, and stay informed [17]. Yet creators rarely verbalize essential visual details, leaving many videos hard to follow [17, 18]. Audio descriptions (AD) make videos accessible by narrating key visual events to complement the existing audio [1, 31]. Unlike extended AD that pauses playback, inline AD preserves the video’s runtime by fitting descriptions into natural pauses (Fig. 1). Effective inline AD thus goes beyond listing what is on screen: it requires temporal precision and content selectivity to build a coherent narrative that enriches, rather than repeats or interrupts, the original audio.

While high-quality audio descriptions are professionally crafted for film and TV [22], it is rarely available for user-generated videos [17], motivating semi-automated [5, 18, 24, 36] and automated generation approaches [3, 8, 15, 16, 32, 34] that have the potential to scale access for millions of BLV audience members. Recent progress in general-purpose multimodal large language models (MLLMs) makes it newly possible to generate AD scripts directly from video. Models such as Gemini 2.5/3.5 [7], GPT-5.2 [21], and Qwen-3.5-Omni [25] can take a video as input and output a timestamped narration script to be rendered via text-to-speech (TTS). Yet are MLLM-generated audio descriptions understandable and enjoyablefor BLV audiences of user-generated videos? Answering this requires evaluating the full end-to-end pipeline: generating the AD script, synthesizing speech via TTS, and measuring how well the rendered audio fits within the natural silences of the original video.

![](images/a33b26fa283559bfde45e989d5cf153270d3e12eb05135d53e54a48704055151.jpg)  
Figure 1: Temporal Constraints of Inline Audio Descriptions. Inline AD must fit precisely into natural silences within a video’s narration (purple). Both the professional “Gold” human AD from MTG-AD (green) and our model-generated “AutoAD” (orange) fit within these tight temporal gaps without overlapping speech. While AutoAD generates vivid, coherent descriptions with timing-aware prompts, the professional AD achieves stronger narrative flow by capturing camera perspectives (e.g., “drone footage”).

Prior work largely evaluates generated AD on trimmed video segments against a single reference [9, 35], through short-clip human ratings, or with coarse LLM scores [15]. Evaluating isolated clips out of context misses sequence-level coherence [14], spoken AD duration, un-narrated “dead air” gaps, and speech overlaps that shape the end-to-end BLV listening experience. Moreover, unlike movies with professional reference AD [27], user-generated videos rarely have expert annotations, forcing prior benchmarks to rely on synthetic [9] or novice-authored proxies where coarse metrics quickly saturate [15].

Our work makes three contributions: First, we establish a high-quality benchmarkfor inline AD on user-generated videos, MTG-AD, comprising professionally crafted ADs for 433 YouTube videos (∼36 hours) co-created by paired blind and sighted AD specialists, alongside outputs from nine frontier MLLMs. Second, we introduce MTG-AD-Eval, an end-to-end, reference-free evaluation framework grounded in expert AD guidelines that pairs TTS-based temporal metrics with multimodal LLM judges, analyzing cross-judge biases and transfer to YouDescribe. Finally, we validate our metrics and benchmark against humanjudgments from sighted raters and 7 BLV participants, showing strong alignment with human ratings, including resolving reference-matching constraints in prior works, and identifying where frontier MLLMs still trail human narrative pacing.

## 2 Related Work

Audio Description Datasets. Most existing AD benchmarks draw from professional descriptions created for movies [8, 9, 27, 29, 35] or TV series [34] (Tab. 1). While these corpora establish what and how to describe in scripted entertainment, user-generated videos vary widely in intent, camera work, editing pace, on-screen text, and speech density, from silent montages to near-continuous narration. Movie AD benchmarks also evaluate models on pre-trimmed 2–20 s clips around known AD timestamps, whereas inline AD for user-generated content requires reasoning over untrimmed, multi-minute videos (averaging 5 minutes in MTG-AD, a horizon shared with long-video benchmark such as Neptune [19]) to decide both when and what to describe. Because professional AD rarely exists for user-generated videos, prior datasets rely on volunteer descriptions of uneven quality on YouDescribe [28], synthetic MLLM descriptions of short ∼10 s clips [15], or repurposed narrations from instructional videos [9]. MTG-AD addresses this gap with 36 hours of full-length user-generated videos co-annotated by paired sighted and blind AD professionals.

<table><tr><td>Dataset</td><td>Video Domain</td><td>Total Duration</td><td>Avg. Clip / Video Len.</td><td>Source of AD</td><td># AD Segments</td><td>Speech / Audio</td><td>Eval. Scope</td></tr><tr><td>M-VAD [27]</td><td>Movies</td><td>84.6 hrs (92 movies)</td><td>6.2s</td><td>Professionals (DVS)</td><td>48,986</td><td>Mixed</td><td>Segment (gaps)</td></tr><tr><td>MAD [29]</td><td>Movies (CLIP feats.)</td><td>1,207.3 hrs (650 movies)</td><td>4.1 s</td><td>Professionals (AudioVault)</td><td>384,600</td><td>Mixed</td><td>Segment (gaps)</td></tr><tr><td>MAD-v2 [8]</td><td>Movies (CLIP feats.)</td><td>892 hrs (488 movies)</td><td>~4.1 s</td><td>Professionals (AudioVault)</td><td>264,000</td><td>Mixed</td><td>Segment (gaps)</td></tr><tr><td>Movie101v2 [35]</td><td>Movies (ZH/EN)</td><td>353 hrs (203 movies)</td><td>11-20s</td><td>Professionals</td><td>71,000</td><td>Non-dialogue</td><td>Segment (gaps)</td></tr><tr><td>CMD-AD [9]</td><td>Movie Clips</td><td>477 hrs (1,432 movies)</td><td>~2 min</td><td>Professionals (AudioVault)</td><td>~101,000</td><td>Mixed</td><td>Segment (gaps)</td></tr><tr><td>TV-AD [34]</td><td>TV Series</td><td>251 hrs (426 episodes)</td><td>2.1 s</td><td>Professionals (AudioVault)</td><td>34,013</td><td>Mixed</td><td>Segment (gaps)</td></tr><tr><td>HowTo-AD [9]</td><td>Instructional</td><td>23,652 hrs (~180k videos)</td><td>Variable</td><td>Synthetic (ASR-adapted)</td><td>3.4M</td><td>Heavy dialogue</td><td>N/A (train only)</td></tr><tr><td>VideoA11y-40K [15]</td><td>User-generated</td><td>- (40k videos)</td><td>~10s</td><td>MLLMs (Synthetic)</td><td>40,000</td><td>Ignored</td><td>Segment</td></tr><tr><td>YouDescribe (YuWA) [28]</td><td>YouTube</td><td>~310 hrs (~5.5k videos)</td><td>~5 min</td><td>Volunteers / Novices</td><td>~76,000</td><td>Mixed</td><td>N/A (archive)</td></tr><tr><td>MTG-AD (Ours)</td><td>User-generated</td><td>36 hrs (433 videos)</td><td>~5 min</td><td>Professional + Blind pair</td><td>8,126</td><td>Mixed</td><td>Full (end-to-end)</td></tr></table>

Table 1: Comparison of Audio Description Datasets. Prior movie and TV benchmarks evaluate isolated silent intervals using pre-trimmed clips or pre-extracted visual features, while prior user-generated or instructional datasets rely on volunteer narrations, short clips with ignored audio, or synthetic ASR adaptations. MTG-AD provides full-length user-generated videos co-annotated by paired sighted and blind AD professionals and evaluates end-to-end inline AD across each entire video.

(a) Duration  
![](images/594e72f1f68af143ac71294b4dec4464b6c115dc488e28e7eaa8f51534e62e27.jpg)  
Duration (min)

(b) AD Rate  
![](images/1d878331447205acc88cd31345d278bd66d29d4705e959706f5781becb431fc5.jpg)  
AD segs / min

(c) Categories  
![](images/d04a08e9aaae3259224a6715bffe0ddd023cb57b77477482a812b22bcfb9dda8.jpg)  
Share of videos (%)

(d) Speech Density  
![](images/7bf5b536b8df66d932553f8ba4ef16f39167f4c0a322a4fe4e3adce0ddcfa538.jpg)  
Speech ratio

(e) Silence Gaps  
![](images/8b5f729d09d1fad28308d41b7071741e80b126ad44f008554e52a9a735093cf3.jpg)  
Gap ≤30s (s)  
Figure 2: Dataset Statistics. MTG-AD comprises 433 professionally annotated videos (∼36 hours): (a) video durations center around 5 minutes; (b) human-annotated AD segments per minute; (c) vlogs form the largest category alongside travel, nature, and education; both (d) speech density and (e) silence-gap distribution illustrate the tight pauses into which inline AD must fit.

Evaluating Audio Descriptions. Automated AD evaluation largely compares generated text against a ground-truth reference [6], using n-gram or embedding similarity metrics such as BLEU [23] and BERTScore [37], or domain-specific metrics for character naming and semantic alignment such as MNScore [35], CRITIC [9], and LLM-AD-Eval [9]. However, single-reference similarity on isolated clips poorly reflects listener comprehension and penalizes valid alternative descriptions. Recent work moves beyond isolated single-reference scoring toward sequence-level coherence [14] and QA-based comprehension checks [13], yet still evaluates text scripts without rendering audio to test whether descriptions fit natural pauses without overlapping speech or leaving long silences.

Human studies with external raters or BLV listeners better capture end-to-end playback quality, though full-video listening studies are time-consuming and typically cover few videos [2, 24]. On short ∼10 s clips, VideoA11y [15] found that guideline-prompted MLLM descriptions surpassed novice captions and matched trained describers, and that LLM-as-a-judge ratings aligned with BLV judgments, echoing broader findings on LLM judges for audio captioning [33]. Still, prior work has not evaluated end-to-end inline AD on full-length user-generated videos, compared MLLM outputs against professionally co-authored AD in that setting, or validated reference-free timing and quality metrics against sighted and BLV listeners.

## 3 Mind the Gap (MTG-AD) Dataset

MTG-AD benchmarks end-to-end inline audio description on user-generated YouTube videos, a domain frequently watched by BLV viewers [15, 17, 28] where professional AD rarely exists (Figs. 1 and 2). The dataset comprises 433 English videos (∼35.9 hours; mean 5.0 min) partitioned into 332 validation videos for prompt and method development and 101 test videos for multi-model evaluation. Each video includes inline AD co-authored by paired sighted and blind AD professionals, a timestamped YouTube ASR transcript (26.3% of videos have no transcribed speech, such as musicbacked montages), and pre-extracted dense visual event lists (D, Sec. 4). For the 101 test videos, we also release unedited AD scripts and per-video metric scores across 12 model–prompt configurations spanning four prompting strategies and nine frontier multimodal models.

## 3.1 Video Selection and Filtering

We sampled 557 candidate English videos around 3–7 minutes long from YouTube channels with at least 1,000 subscribers across diverse user-generated categories, led by vlogs (46%), education (16%), cooking, nature, travel, and science in the final set (Fig. 2c). The professional AD teams reviewed every candidate in full to assess whether inline AD could be added without pausing playback, filtering candidates on two criteria: (i) Temporal viability: videos with nonstop dialogue or narration, or vlogs whose on-screen subtitles left no audio space for additional descriptions, were marked nonviable, though annotators revisited borderline cases to include partial descriptions where brief pauses permitted; and (ii) Topical diversity: after annotating 3–5 videos of a recurring style (e.g., overhead cooking tutorials), further near-duplicates were skipped. Of the 557 candidates, 438 (78.6%) were annotated, 92 (16.5%) were excluded as non-viable, and 27 (4.8%) were skipped for redundancy. Five duplicate candidates were annotated twice; retaining the first annotation yields 433 unique videos totaling 35.9 hours (Fig. 2a,d).

![](images/035506ead02919598d75c7c5d91e405cbd0bb99a3d0b1b8298365fe92ced803f.jpg)  
Figure 3: Prompt strategy comparison on content and temporal metrics (Sec. 4) for Gemini-2.5 Flash. Content (first two panels; higher is better): guideline prompts improve visual prioritization and on-screen text recall over Simple AD. Temporal precision (last three panels; lower is better): Simple AD leaves long silences, whereas Guideline only overfills gaps and overlaps speech; Gd. + Timing ASR, which adds the timestamped transcript and per-gap word budgets, best balances content against timing.

## 3.2 Professionally Co-Authored AD

Descriptions were produced by two two-person professional teams, each pairing one sighted and one blind AD writer so that every script reflects both visual detail and BLV listener priorities. For each video, the sighted writer first watched the full video to establish narrative context and pacing, then rewatched it in segments to draft concise, timestamped descriptions of key visual events and on-screen text not already conveyed by speech or sound effects. Next, the writer imported the script into an audio-editing environment against the original soundtrack, adjusting phrasing and start times so each spoken description fit cleanly into natural pauses without overlapping dialogue. Finally, the blind writing partner reviewed the timed playback end-to-end to audit narrative clarity and standalone comprehension before sign-off. Across all 433 videos, MTG-AD contains 8,126 AD segments (6,214 validation, 1,912 test), with a mean of 18.8 (median 16, IQR 10–26) segments per video, 17.0 words per segment (median 15), and a median rate of 3.4 segments per minute (IQR 2.0–5.7; Fig. 2b).

## 3.3 Eliciting Model-Generated AD on Untrimmed Videos

Unlike trimmed-clip captioning with pre-segmented gaps, generating inline AD on untrimmed videos requires a model to select salient visual events while aligning to speech pauses and fitting spoken AD within each gap. To study how well MLLMs handle this end-to-end task and what context or reasoning steps resolve their failure modes, we designed four prompts that we studied on Gemini-2.5 Flash. Simple AD (App. Fig. 17) tests minimal-instruction capability, requesting only timestamped descriptions during non-speech intervals; Guideline only (App. Fig. 18) adds expert AD rules (e.g., prioritizing key visuals, reading on-screen text verbatim, remaining objective, and omitting audible cues), testing whether domain guidelines suffice when speech pauses must be inferred directly from video; Guideline + ASR (App. Fig. 19) supplies the timestamped speech transcript (start and end times) and a 1–3 words/s speaking-rate target to ground pause boundaries in text; and Gd. + Timing ASR (App. Fig. 20) decomposes temporal planning into explicit steps: computing each silent gap’s duration, capping length at 3 words per silent second (at most three sentences), and skipping details when space is insufficient. To compare model families under our strongest elicitation setup (Gd. + Timing ASR), on the 101 test videos we pair all four Gemini-2.5 Flash prompt runs with eight additional frontier models: GPT-4.1 mini, GPT-5.2, Claude 4, Claude 4.5, Qwen-2-VL-Max, Gemini-3.5 Flash, and Qwen-3.5-Omni-Flash/Plus. Providing the timestamped transcript alongside the video gives both native audio-video models (Gemini-2.5/3.5 Flash, Qwen-3.5-Omni) and vision-and-text models identical speech content and silence boundaries without requiring raw audio ingestion.

## 4 MTG-AD-Eval: Evaluation Framework

From expert guidelines to metrics. Following prior work [15,24], we reviewed five expert AD guidelines chosen for breadth and comprehensiveness: the American Council of the Blind [1], DCMP’s Description Key [4], ADLab’s film guide [26], W3C’s WCAG [30], and YouDescribe’s tips [28]. We also reviewed studies of BLV audiences’ perceptions of AD to find where audience preferences depart from prescribed rules [11, 12, 17, 20]. Together with researchers in HCI, accessibility, and computer vision, we distilled these to ten guidelines covering content (G1–G4), style (G5–G6), and audio (G7–G10); Appx. Sec. A lists them all. In a pilot study, BLV audience members reviewed AD generated with a basic prompt and named the problems that most hurt their experience: AD talking over speech, long unexplained silences, and descriptions out-of-sync. We therefore turn the six guidelines tied to these problems (G1.1–G1.3, G2, G7, G8) into automatic metrics, and add the VideoA11y rubrics [15] to rate writing quality. Tab. 2 summarizes the metrics.

Table 2: Evaluation metrics at a glance. Metrics are computed per video and averaged across videos. Scorer: Prog. = programmatic or word arithmetic; LLM = text-only LLM judge; MLLM = video + text judge (prompts in Appx. Sec. J). Ref-free: runs without human AD. D = dense visual information; $\mathrm { T T S } = \mathrm { A D }$ rendered at 1.2×.
<table><tr><td>Metric (guideline)</td><td>Question it answers → computation</td><td></td><td>Scorer Ref-free</td><td>Better</td></tr><tr><td colspan="5">Content metrics</td></tr><tr><td>Prioritization (G1.1)</td><td>Is AD spent on salient visual events? → Judge scores each segment as ci: covered (1), partly (0.5), or not (0) for support in  $\mathcal { D } ;$  mean č.</td><td>LLM</td><td>√</td><td>↑[0, 1]</td></tr><tr><td>No Redundancy (G1.2)</td><td>Does AD avoid restating audible cues? → Judge scores each segment&#x27;s overlap with nearby transcript or explicit sound cues,  $r _ { i } \in [ 0 , 1 ] ; 1 - \bar { r } .$ </td><td>LLM</td><td>√</td><td>↑[0, 1]</td></tr><tr><td></td><td>On-Screen Text (G1.3) Is on-screen text conveyed? → Fraction of on-screen text words from D in the combined AD and ASR vocabulary.</td><td>Prog.</td><td>√</td><td>↑ [0, 1]</td></tr><tr><td colspan="5">Temporal metrics</td></tr><tr><td>Timeliness (G2)</td><td>Is each description heard near its visual event? → Judge matches each AD segment to an event at  $t _ { i }$  in D; mean  $| s _ { i } - t _ { i } |$  on matches.</td><td>LLM</td><td>V</td><td>↓ sec.</td></tr><tr><td>Audio Overlaps (G7)</td><td>Does AD talk over speech? → Total duration  $| A \cap S |$  where TTS-rendered AD and speech play simultaneously.</td><td>Prog.</td><td>√</td><td>↓ sec.</td></tr><tr><td>Audio Coverage (G8)</td><td>How much available silence does AD use? -  $ | A | / ( T$  - |S|); values &gt;1 indicate AD exceeds total non-speech time.</td><td>Prog.</td><td>√</td><td>↑(≤ 1)</td></tr><tr><td>Avg. Silence (G8)</td><td>How long are the remaining quiet stretches? → Mean duration of gaps in  $[ 0 , T ] \setminus ( A \cup S )$  with neither speech nor AD.</td><td>Prog.</td><td>√</td><td>↓ sec.</td></tr><tr><td colspan="5">Holistic script quality (VideoA11y rubric: Descriptive, Objective, Accurate, Clear; 4 scores)</td></tr><tr><td>VideoA11y-Ref [15]</td><td>How closely does AD match human AD? → Judge rates the script on each dimension against the human-written reference.</td><td>LLM</td><td>X</td><td>↑1-5</td></tr><tr><td>Quality-RF (ours)</td><td>How well does AD describe the video? → Same four ratings, judged MLLM directly against the video and its ASR transcript.</td><td></td><td>L</td><td>↑1-5</td></tr></table>

End-to-end setup and notation. We score the AD of the whole video as a listener would hear it, not isolated clips. For a video of duration T (seconds), a system outputs an AD script $\{ ( s _ { i } , x _ { i } ) \} _ { i = 1 } ^ { N }$ with segment start times $s _ { i }$ and texts $x _ { i }$ . We synthesize each $x _ { i }$ with an off-the-shelf text-to-speech (TTS) engine at a fixed $1 . 2 \times$ speaking rate to obtain its spoken duration $d _ { i }$ and playback interval $I _ { i } = [ s _ { i } , s _ { i } + d _ { i } ]$ . Let $A = \textstyle \bigcup _ { i = 1 } ^ { N } I _ { i }$ denote the union of AD intervals and | · | the total duration of a set of intervals, so $| A |$ is the total time AD plays. The metrics use two precomputed inputs per video that require no human-written AD: (i) the timestamped ASR transcript, yielding both spoken words and speech intervals $[ a _ { j } , b _ { j } ]$ with union $\textstyle S = \bigcup _ { i } [ a _ { j } , \mathbf { \hat { b } } _ { j } ] ;$ ; and (ii) dense visual information D, a timestamped list of the video’s important visual events and on-screen text, generated by Gemini-2.5 Flash (prompt in App. Fig. 16). Combined with Quality-RF, which judges scripts directly against the video and transcript, every metric except VideoA11y-Ref runs on unseen videos without human references. We used Gemini-2.5 Flash as the default judge (and switch to Gemini 3.5 Flash in the released code) and analyze cross-judge sensitivity in Secs. 5 and 5.2.

## 4.1 Content Metrics: What the AD Says

Prioritization (G1.1) asks whether the AD spends its limited budget on visual information needed to follow the video. Given the AD script and D, a judge scores each segment with $c _ { i } \in \{ 0 , 0 . 5 , 1 \}$ for whether $x _ { i }$ describes salient content in D (App. Fig. 21). We report ${ \frac { 1 } { N } } \sum _ { i } c _ { i }$ , which measures segment-level precision against D. It penalizes trivial or invented details but not omissions (measured partly in onscreen-text) or under-description (captured by Coverage and Avg. Silence). No Redundancy (G1.2) checks that AD does not restate what a listener can already hear, such as dialogue or an obvious sound (e.g., door slam). We interleave the transcript and the AD by time, and a judge scores each segment’s redundancy $r _ { i } \in \{ 0 , 0 . 5 , 1 \} ( \mathrm { A p p . } \mathrm { F i g . } 2 3 )$ ). We report $\begin{array} { r } { 1 - \frac { 1 } { N } \sum _ { i } r _ { i } . } \end{array}$ , so higher is better. On-Screen Text (G1.3) measures whether text shown in the video (signs, titles, captions) is conveyed to the listener. Let W be the words of the on-screen text listed in D and V the set of words spoken in the AD or the transcript. The score is the fraction of W found in V, which is a word-level recall computed programmatically (an initial LLM-judged version is shown in App. Fig. 22). Videos without on-screen text are excluded.

![](images/ef78fa47348ac56a12305287dcc1c4cf463250bf9da3884568f58b8e0921bd91.jpg)  
Figure 4: Frontier model comparison on content (G1.1, G1.3; higher is better) and temporal metrics $( \mathrm { G } 2 , \mathrm { G } 7 ,$ Avg. Silence; lower is better), all with the Gd. + Timing ASR prompt. Bars show means with 95% CIs over the 101 test videos. Models marked <sup>†</sup>: G1.1 and G2 scored by a Gemini-3.5 Flash judge (others: Gemini-2.5 Flash). App. Tabs. 5 and 6 show detailed scores.

## 4.2 Temporal Metrics: When and How Much the AD Speaks

Timeliness (G2) measures how far each description is from the moment its event appears. A judge pairs each AD segment’s start time s<sub>i</sub> with the event in D it describes, occurring at time $t _ { i \cdot }$ , or marks it unmatched (App. Fig. 24). Over the set M of matched segments, we report Timeliness in seconds (unmatched segments are penalized by G1.1). While Timeliness evaluates when AD starts, fitting inline AD into pauses depends on the rendered speech interval $I _ { i }$ so it shouldn’t spill into the next dialogue. Audio Overlaps (G7) is the total time during which AD and speech play together, ${ \cal T } _ { \mathrm { o v e r l a p } } = \bar { | } A \cap { \cal S } |$ . Audio Coverage (G8) compares how long the AD speaks relative to how much silence is available $\mathrm { C o v e r a g e } = | \tilde { A | } / T - | S |$ . This checks if the AD covers most of the video without filling every pause, so values somewhat below 1 are desired. We deliberately do not clip the ratio. A value above 1 means rendered AD exceeds all available silence combined $( e . g .$ , 1.23 for the Guideline only prompt), so it flags over-generation. These three metrics are computed as:

$$
\begin{array} { r } { \mathrm { T i m e l i n e s s } = \frac { 1 } { | M | } \sum _ { i \in M } | s _ { i } - t _ { i } | ; \quad T _ { \mathrm { o v e r l a p } } = \big | \bigcup _ { i } I _ { i } \cap \bigcup _ { j } [ a _ { j } , b _ { j } ] \big | ; \quad \mathrm { C o v e r a g e } = \frac { \big | \bigcup _ { i } I _ { i } \big | } { T - \big | \bigcup _ { j } [ a _ { j } , b _ { j } ] \big | } . } \end{array}
$$

Avg. Silence (G8) measures the dead air left after the AD is inserted. We take the gaps in $[ 0 , T ]$ during which neither speech nor AD plays, including those at the start and end, and report their mean length in seconds. Coverage and Avg. Silence complement each other: a script can reach high Coverage with a few long segments yet leave long silences elsewhere.

## 4.3 Holistic Script Quality

Following VideoA11y [15], an LLM judge rates the full AD script on a likert scale (1–5) on four dimensions: Descriptive (vivid, relevant visual detail), Objective (reports what is visible without speculation), Accurate (visual details are correct), and Clear (easy to follow without seeing the video). We evaluate two variants:

• VideoA11y-Ref (App. Fig. 25) is the original reference-based setup: the judge compares the generated AD with the human-written AD. A 5-minute video admits many valid descriptions, so its Accurate score effectively measures Reference Alignment, which penalizes correct details that the human describer happened not to mention. Human AD scored this way is compared with itself and therefore serves only as an upper bound.

• Quality-RF (ours, App. Fig. 26) removes the reference human AD. The judge is given both the raw video (frames and audio) and the ASR transcript to ground spoken dialogue explicitly (input ablations in Appx. Sec. E); it summarizes the video, explains its reasoning, and then scores the same four dimensions. This avoids the reference-matching penalty and works without human AD.

## 5 Experiments and Results

We evaluate (1) prompting strategies and frontier MLLMs with our automated metrics, (2) the reliability of LLM judges and transfer to the YouDescribe dataset [28], and (3) human ratings from sighted raters and BLV participants alongside agreement with our metrics. Each value is a mean over videos with a 95% confidence interval. Full tables are shown in Appx. Sec. B.

## 5.1 Metric-based Evaluations

Metric-guided prompt refinement. Scoring the four Gemini-2.5 Flash prompts across validation videos (Fig. 3, App. Tab. 4) illustrates how MTG-AD-Eval diagnoses failure modes to guide prompt design. Simple AD under-describes the video (Coverage 0.41, Avg. Silence 17.95 s, On-Screen Text 0.47). Adding domain rules in Guideline only raises Prioritization (0.79) and On-Screen Text (0.69), yet MTG-AD-Eval exposes severe over-generation (1.23 Coverage, 59.19 s of speech overlap) that content metrics alone would miss. With Guideline + ASR which includes explicit time spans of dialogue overlaps reduce to 21.34 s. Further emphasizing per-gap duration and wordbudget constraints in Gd. + Timing ASR reduces overlaps considerably to 12.77 s while preserving Prioritization (0.76). Human AD remains better timed (7.51 s overlap, 4.19 s Avg. Silence); its slightly lower Prioritization (0.73) arises because D itself is MLLM-extracted, leaving fine-grained human-observed actions (e.g., a truck braking or swerving in App. Fig. 15) and brief pause-fitted beats (e.g. A seal blinking) uncredited.

![](images/019cb4a826c805965ce58db8d2ef5cb85d30c0b35e6444fee9295ae24654b0bb.jpg)  
Figure 5: Qualitative alignment (VideoA11y-Ref) ratings on the same videos as Fig. 4. Because the judge compares against the human AD, Accurate here measures agreement with that reference (Reference Alignment). Models marked <sup>†</sup> were scored by a Gemini-3.5 Flash judge and the others by Gemini-2.5 Flash, so scores should be compared only within each group.

Frontier models. We score nine models on the 101 test videos (Fig. 4, App. Tabs. 5 and 6). Native audio-video models (and more recent versions) lead on Prioritization: Qwen-3.5 and Gemini models have scores between 0.76–0.94, versus 0.51–0.59 for the others. Timeliness follows a similar split: with the recent Qwen-3.5 and Gemini-3.5 models describing events within 2−5 s of when they appear and their previous (Qwen-2, Gemini-2.5) versions stay close (within 5 − 7 s), whereas Claude 4/4.5, GPT-5.2, and GPT-4.1 mini drift 20–65 s away. The programmatic timing metrics reveal pause-budget trade-offs among the strongest models. Gemini-3.5 Flash nearly eliminates overlap (0.26 s) by having shorter descriptions (Coverage 0.45) and achieves the shortest Avg. Silence (4.85 s); Gemini-2.5 Flash balances higher silence use (Coverage 0.74) with some overlap (8.92 s; 6.63 s Avg. Silence); and the Qwen models fill more silence (Coverage 0.72–0.94) at the cost of 30.45–52.14 s of overlaps. No model leads on every metric; On-Screen Text, for example, stays within 0.56–0.64 across all nine.

Holistic Script Quality. Models scored with the Gemini-2.5 Flash judge (Fig. 5) average 2.0–3.3 on Descriptive, Objective, and Clear (2.2–3.3) yet only 1.16–1.63 on Reference Alignment (Accurate). The † models, scored by Gemini-3.5 Flash, score higher. Since the judge also changed, part of this jump may come from the judge rather than the generator. Low Reference Alignment partly reflects the reference-matching penalty described in Sec. 4, which penalizes valid visual details omitted by the human describer and which Quality-RF resolves against human ratings in Sec. 5.3.

![](images/3c8a53ebcfac5dcc339a1edd76d55ed92cb3b4483ae20a12f4b6edb59f324007.jpg)

![](images/f8d452c8aacf6ce8fd40a11b461fd985307c32c767eb9b1325d12164a6f662c2.jpg)  
Figure 6: Judge Alignment and Bias. (a, left) Inter-judge Pearson correlation across 101 test videos. Each score is the mean across five metrics Prioritization (G1.1) and VideoA11y-Ref rubrics; GPT-4.1 mini and Qwen-2-VL-Max align most closely. (b, right) Deviation from model-specific mean scores. Shows systematic bias of each judge by calculating the mean deviation from the global average score per video. Positive values (green) indicate "lenient" judging, while negative values (pink) indicate "harsh" judging relative to the consensus.

![](images/8dd8b0d543cf03418d869387b039e792a65d28d0cefd8636852a55dc3305e50b.jpg)  
Figure 7: YouDescribe generalization. (a): Mean values (with 95% CI) for Prioritization, On-Screen Text coverage, Timeliness, and Avg. Silence across 2 available human AD sources and our automated AD on the YouDescribe dataset. (b) Distribution of available silence gaps (right) shows significantly longer gaps compared to our dataset due to the source videos being largely silent or lacking pre-existing dialogue.

## 5.2 LLM as Judge and Generalization

Inter-judge reliability. As observed with judgments of script quality between Gemini 2.5 Flash and 3.5 Flash judges, unconstrained Likert scores shift with judge leniency, so some changes and jumps conflates generator gains with the judge change. To test sensitivity to the evaluator, we re-score the same ADs with Claude, Gemini, GPT and Qwen judges and correlate their per-video scores on five LLM-judged metrics (Fig. 6a) All judge pairs correlate positively $( r = 0 . 3 8 – 0 . 7 4 )$ , and GPT-4.1 mini and Qwen-2-VL-Max agree most. However, calibration differs across judges (Fig. 6b): some judges are systematically more lenient (e.g., GPT-4.1 mini) and others harsher (e.g., Gemini-2.5-Flash), but these offsets are small. This supports using MLLM judges to rank systems, provided all systems in a comparison use the same judge.

Generalization to YouDescribe. When applying our evaluation suite to external YouDescribe videos (Fig. 7), we observe that “Premium” human AD trends ahead of “Standard” volunteer AD on Timeliness (2.16 s vs. 2.85 s) and Prioritization (0.82 vs. 0.80), though CIs overlap. This suggests that the metrics transfer to other user-generated videos and AD sources.

## 5.3 Sighted and BLV Participant Studies & Metric Validation

We conducted complementary human studies with sighted raters and BLV participants using a custom playback interface. Sighted raters and BLV listeners serve distinct, complementary roles in evaluating AD [10]: sighted raters can directly inspect the video frames to verify visual-to-text faithfulness, missing visual events, and exact visual-timestamp alignment across many model/prompt pairs, whereas BLV participants evaluate whether the audio track and descriptions form a coherent, non-intrusive, and engaging standalone narrative without visual context.

Sighted Rater Study. A pool of 8 sighted raters evaluated 42 videos, with AD instances spanning multiple models, prompts, and professional Human AD, across 11 dimensions including Coverage, Accuracy, Relevance, Clarity, Timeliness, and Audio Issues on a likert scale (full questionnaire in Appx. Sec. F; and ratings of prompt comparison are in App. Fig. 10). Inter-annotator agreement (Krippendorff’s α, App. Tab. 7) across all metrics was 0.654, led by Timeliness (0.721) and Audio Issues (0.700). Comparing models (Fig. 8), Gemini-2.5 Flash with Gd. + Timing ASR prompt approaches human ground truth in Accuracy and Timeliness but lags on audio overlap issues.

![](images/a28b0b966bb2db51acab05b74771e67264bef41283abbaeb511fc2f264dc2741.jpg)  
Figure 8: Sighted rater model comparisons. Mean Likert scores across five (of 11) key dimensions on videos evaluated by sighted raters (↑ indicates higher is better). Gemini-2.5 Flash approaches Ground Truth on Accuracy, Timeliness and script Quality, while Claude 4 has fewer Audio Issues due to lower speech coverage.

![](images/7ac4837bdbfc8276e88932ce563f7ce8668268c9dfac532b5299feccb23aac20.jpg)  
Figure 9: BLV study. Mean Likert scores across 8 key dimensions (error bars depict 95% confidence intervals). BLV participants evaluated videos played with professional human written ADs (MTG-AD) and AutoAD (Gemini-2.5 Flash with Gd. + Timing ASR). They rated both AD versions consistently high across all metrics and the human generated AD slightly consistently better in all dimensions. (Full likert scoring in App. Fig. 12)

Table 3: Empirical validation of automated metrics against Sighted (n = 42) and BLV (n = 28) human judgments. Spearman rank correlations (ρ) and p-values across our Reference-Free Guideline Metrics (G1– G8), Reference-Based VideoA11y-Ref, and Reference-Free Quality-RF (video + ASR). Full qualitative interpretations and 4-variant ablations are in Appx. Sec. E.
<table><tr><td>Category</td><td>Automated Metric</td><td>Cohort</td><td>Target Human Dimension</td><td>Spearman ρ</td><td>p-value</td></tr><tr><td rowspan="6">Reference-Free Guideline Metrics</td><td>Audio Overlaps (G7) Avg. Silence (G8)</td><td>Sighted (n = 42) Sighted (n = 42)</td><td>Audio Issues (No Overlaps) Gaps of Silence (Few Quiet Gaps)</td><td>-0.467 -0.415</td><td>0.0018** 0.0063**</td></tr><tr><td>Timeliness (G2)</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>Sighted (n = 42)</td><td>Timeliness</td><td>-0.228</td><td>0.1470</td></tr><tr><td>Prioritization (G1.1)</td><td>Sighted (n = 42)</td><td>Coverage</td><td>+0.246</td><td>0.1160</td></tr><tr><td>No Redundancy (G1.2)</td><td>BLV (n = 28)</td><td>Consistency / Non-Redundancy</td><td>+0.386</td><td>0.0427*</td></tr><tr><td>Audio Overlaps (G7)</td><td>BLV (n = 28)</td><td>Audio Quality (truncated in UI)</td><td>+0.132</td><td>0.5024</td></tr><tr><td rowspan="6">Reference-Based VideoA11y-Ref [15]</td><td>VideoA11y-Ref Clear</td><td>Sighted (n = 42)</td><td>Clarity</td><td>+0.453</td><td>0.0026**</td></tr><tr><td>VideoA11y-Ref Descriptive</td><td>Sighted (n = 42)</td><td>Descriptive</td><td>+0.328</td><td>0.0338*</td></tr><tr><td>VideoA11y-Ref Clear</td><td>BLV (n = 28)</td><td>Clarity</td><td>+0.388</td><td>0.0411*</td></tr><tr><td>VideoA11y-Ref Descriptive</td><td>BLV (n = 28)</td><td>Descriptive</td><td>+0.340</td><td>0.0771</td></tr><tr><td>VideoA11y-Ref Objective</td><td>BLV (n = 28)</td><td>Objective</td><td>+0.335</td><td>0.0811</td></tr><tr><td>VideoA11y-Ref Accurate</td><td>BLV (n = 28)</td><td>Perceived Accuracy</td><td>-0.264</td><td>0.1743</td></tr><tr><td rowspan="4">Reference-Free Quality-RF (Video + ASR, Ours)</td><td>Quality-RF Descriptive</td><td>BLV (n = 28)</td><td>Descriptive</td><td>+0.583</td><td>0.0011**</td></tr><tr><td>Quality-RF Clear</td><td>BLV (n = 28)</td><td>Clarity</td><td>+0.415</td><td>0.0281*</td></tr><tr><td>Quality-RF Objective</td><td>BLV (n = 28)</td><td>Objective</td><td>+0.365</td><td>0.0561</td></tr><tr><td>Quality-RF Accurate</td><td>BLV (n = 28)</td><td>Perceived Accuracy</td><td>+0.244</td><td>0.2109</td></tr></table>

BLV Participant Study. We recruited 7 BLV participants who regularly watch YouTube. Each participant evaluated four videos randomly sampled from our dataset (n = 28 video-level evaluations), listening to both AutoAD (Gemini-2.5 Flash with Gd. + Timing ASR) and the professionally authored Human AD. Participants were unaware of the AD source and we randomized and counterbalanced the order of the AD sources. Participants provided ratings for each AD (Fig. 9), their preference between the two AD (Appx. Fig. 12), and qualitative feedback about each AD. Participants rated both versions consistently high across all dimensions (Fig. 9). While participants marginally preferred humanauthored AD (53.6%, App. Fig. 12), Wilcoxon Signed-Rank tests showed statistically significant differences only for overall Satisfaction (p < 0.05) and perceived Accuracy $( p < 0 . 0 5 )$ , with nearparity on Quality, Clarity, and Objectivity. Individual preferences varied markedly across content genres (Sec. D.1): several participants preferred the denser visual detail of AutoAD for actionor travel-oriented clips while favoring the concise narrative pacing of professional human AD for dialogue-heavy vlogs.

Do the automated metrics agree with people? We compute video-level Spearman rank correlations between each metric and the matching human rating for both cohorts (Tab. 3; full results in Appx. Sec. E). For sighted raters, each data point is a rated video with its AD. For BLV listeners, each data point is one participant’s rating of the AutoAD version of a video $( n = 2 8 )$ , paired with that AD’s metric scores. Negative ρ is expected for metrics measured in seconds, since fewer seconds of overlap or silence should mean a higher 1–5 rating. For sighted raters, the two programmatic timing metrics are the strongest predictors: Audio Overlaps (G7) tracks ratings of audio issues and Avg. Silence (G8) tracks ratings of quiet gaps (both $p < 0 . 0 1 )$ . VideoA11y-Ref Clear and Descriptive also correlate significantly with the matching ratings. For BLV listeners, No Redundancy (G1.2) and

VideoA11y-Ref Clear are significant predictors. Audio Overlaps correlates only weakly with BLV ratings of audio quality, by design of the study: our video player cut off any AD segment as soon as speech began. BLV listeners therefore never heard AD collide with speech; AD that was too long instead showed up as abrupt cutoffs, which listeners penalized under Clarity.

Resolving reference-matching bias with Quality-RF. VideoA11y-Ref Accurate correlates negatively with BLV ratings of accuracy $( \rho = - 0 . 2 6 4 )$ . Two factors explain this. First, listeners without visual access judge accuracy by whether the narration is plausible and consistent [10]. Second, the reference-based score penalizes correct details that the human describer omitted. Scoring against the video and transcript with Quality-RF eliminates this penalty, flipping Accuracy positive $( \rho = + 0 . 2 4 4 )$ and substantially strengthens agreement on Descriptive $( \rho = + 0 . 5 8 3 , p = 0 . 0 0 1 )$ and Clear $( \rho = + 0 . 4 1 5 , p = 0 . 0 2 8 )$ . Together, the guideline metrics and Quality-RF form a fully reference-free evaluation suite that is validated against human judgments and works on unseen videos.

## 6 Conclusion and Broader Impact

While frontier MLLMs approach human AD in visual descriptiveness, there’s still a clear gap in temporal precision, remaining silence, and narrative pacing for inline AD. Through the MTG-AD dataset with 36 hours of high-quality ground truth, co-created by teams of blind and sighted professionals, and MTG-AD-Eval, an end-to-end reference-free evaluation suite, we demonstrate:

• Evaluation-Guided Prompting for Precision: The framework allows development of prompts and harnesses for improved audio description generation. We find that incorporating ASR transcripts and temporal reasoning (via the Gd. + Timing ASR strategy) is essential for balancing content richness with strict temporal constraints, reducing audio overlaps.

• Model Performance Benchmarking: Newer-generation omni-modal models (Gemini-3.5 Flash, Qwen-3.5-Omni) raise content prioritization from 0.76 to 0.79–0.94, and Gemini-3.5 Flash cuts speech overlaps to 0.26 s. Still, no model leads on every metric: the models trade overlap against coverage of the silences, and human AD remains better paced.

• User Utility and Reference-Free Evaluation: Our study with blind and low-vision (BLV) participants confirms that while human-crafted AD remains the gold standard for narrative flow, MLLM-generated AD is already highly beneficial for comprehension, and MTG-AD-Eval, our validated reference-free metric suite MTG-AD-Eval (G1.1–G1.3, G2, G7, G8 and Quality-RF) correlates substantially with both sighted and BLV human judgments.

Guidance for System Designers. Inline AD is a multi-objective constrained problem, and a single aggregate score hides failure modes (e.g., high Prioritization with heavy overlaps). For model selection we recommend two stages: (1) Temporal Feasibility Filtering, a hard threshold on Audio Overlaps (G7) so AD does not mask dialogue, and Audio Coverage $( G \& ) \le 1$ to rule out overgeneration (this still leaves some gaps undescribed, as Avg. Silence shows); then (2) Content & Narrative Optimization, maximizing Prioritization (G1.1), No Redundancy (G1.2), and Reference-Free Clarity/Descriptiveness (Quality-RF) among temporally feasible candidates.

Limitations. Our automated metrics cover six guidelines from our BLV pilot (G1.1–G1.3, G2, G7, G8), leaving others such as active voice (G5) and prosody (G10) to future work; G1.1 and G2 also depend on MLLM-extracted events (D) and judge calibration, which can under-credit subtle human-observed details. Second, our BLV study (7 participants, 28 full-length video evaluations) gives rich paired comparisons, but larger cohorts across more genres would better capture individual preferences. Finally, our temporal metrics render AD with a standard TTS engine at a fixed 1.2× speaking rate, and absolute overlap and coverage values depend on this rate. We leave the speaking rate to future work; faster or adaptive per-gap speech could fit richer descriptions into the silences.

Dataset and Code Release. We release MTG-AD under CC-BY-4.0 (hf.co/datasets/google/mind-thegap-inline-ad) and MTG-AD-Eval under Apache-2.0 (github.com/google/mtg-ad). Bundling human and model ADs, ASR transcripts, and pre-extracted visual events (D) to support reproducibility, re-use and further development of the evaluation frameworks.

Broader Impact. The MTG-AD dataset and MTG-AD-Eval reference-free evaluation framework can support automated auditing and generation of inline AD for user-generated tutorials, educational content, and vlogs at scale, and could serve as video accessibility indicators for BLV audiences. It also carries risks: generated AD can hallucinate details that mislead BLV users or misrepresent the creator’s intent, which listeners cannot easily verify, so AI-generated AD should be labeled as such

## Acknowledgements

We thank Cindy Bennett and Shaun Kane for their valuable feedback on the audio description guidelines, and Aidan Bryant for help with an early BLV pilot survey. We also thank Philip Nelson and Chandralekha Motati for their support of this work, as well as the audio description professionals, sighted raters, and blind and low-vision study participants for their annotations and insights.

## References

[1] ACB. American council of the blind, audio description project, guidelines for audio describers. https://www.acb.org/adp/guidelines.html, 2020.

[2] Carmen J Branje and Deborah I Fels. Livedescribe: can amateur describers create high-quality audio description? Journal of Visual Impairment & Blindness, 106(3):154–165, 2012.

[3] Peng Chu, Jiang Wang, and Andre Abrantes. Llm-ad: Large language model based audio description system. arXiv preprint arXiv:2405.00983, 2024.

[4] DCMP. Description key. https://dcmp.org/learn/captioningkey/624, 2020.

[5] Langis Gagnon, Samuel Foucher, Maguelonne Heritier, Marc Lalonde, David Byrns, Claude Chapdelaine, James Turner, Suzanne Mathieu, Denis Laurendeau, Nath Tan Nguyen, and et al. Towards computer-vision software tools to increase production and accessibility of video description for people with vision loss. Universal Access in the Information Society, 8(3):199–218, 2009.

[6] Yingqiang Gao, Lukas Fischer, Alexa Lintner, and Sarah Ebling. Audio description generation in the era of llms and vlms: A review of transferable generative ai technologies. Findings ofthe Association for Computational Linguistics: NAACL 2025, pages 471–490, 2025.

[7] Google DeepMind. Gemini 2.5: Enhanced reasoning and multimodal capabilities. Technical report, Google, June 2025. Gemini 2.5 Technical Report.

[8] Tengda Han, Max Bain, Arsha Nagrani, Gül Varol, Weidi Xie, and Andrew Zisserman. Autoad: Movie description in context. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18930–18940, 2023.

[9] Tengda Han, Max Bain, Arsha Nagrani, Gül Varol, Weidi Xie, and Andrew Zisserman. Autoad iii: The prequel-back to the pixels. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18164–18174, 2024.

[10] Mina Huh, Yi-Hao Peng, and Amy Pavel. Long-form video question answering by blind and low vision users. In Conference on Language Modeling (COLM), 2024. arXiv:2408.06303.

[11] Lucy Jiang, Crescentia Jung, Mahika Phutane, Abigale Stangl, and Shiri Azenkot. “it’s kind of context dependent”: Understanding blind and low vision people’s video accessibility preferences across viewing scenarios. In Proceedings of the 2024 CHI Conference on Human Factors in Computing Systems, pages 1–20, 2024.

[12] Lucy Jiang, Mahika Phutane, and Shiri Azenkot. Beyond audio description: Exploring 360° video accessibility with blind and low vision users through collaborative creation. In Proceedings ofthe 25th International ACM SIGACCESS Conference on Computers and Accessibility (ASSETS ’23), 2023.

[13] Divy Kala, Eshika Khandelwal, and Makarand Tapaswi. What you see is what you ask: Evaluating audio descriptions. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, 2024.

[14] Eshika Khandelwal, Junyu Xie, Tengda Han, Max Bain, Arsha Nagrani, Andrew Zisserman, Gül Varol, and Makarand Tapaswi. More than a moment: Towards coherent sequences of audio descriptions. arXiv preprint arXiv:2510.25440, 2025.

[15] Chaoyu Li, Sid Padmanabhuni, Maryam S Cheema, Hasti Seifi, and Pooyan Fazli. Videoa11y: Method and dataset for accessible video description. In Proceedings of the 2025 CHI Conference on Human Factors in Computing Systems, pages 1–29, 2025.

[16] Kevin Lin, Faisal Ahmed, Linjie Li, Chung-Ching Lin, Ehsan Azarnasab, Zhengyuan Yang, Jianfeng Wang, Lin Liang, Zicheng Liu, Yumao Lu, et al. Mm-vid: Advancing video understanding with gpt-4v (ision). arXiv preprint arXiv:2310.19773, 2023.

[17] Xingyu Liu, Patrick Carrington, Xiang ’Anthony’ Chen, and Amy Pavel. What makes videos accessible to blind and visually impaired people? In Proceedings of the 2021 CHI Conference on Human Factors in Computing Systems, pages 1–4, New York, NY, USA, 2021. ACM.

[18] Xingyu Liu, Ruolin Wang, Dingzeyu Li, Xiang’Anthony’ Chen, and Amy Pavel. Crossa11y: Identifying video accessibility issues via cross-modal grounding. In UIST 2022, 2022.

[19] Arsha Nagrani, Mingda Zhang, Ramin Mehran, Rachel Hornung, Nitesh Bharadwaj Gundavarapu, Nilpa Jha, Austin Myers, Xingyi Zhou, Boqing Gong, Cordelia Schmid, Mikhail Sirotenko, Yukun Zhu, and Tobias Weyand. Neptune: The long orbit to benchmarking long video understanding. In arXiv preprint arXiv:2412.09582, 2024.

[20] Rosiana Natalie, Ruei-Che Chang, Smitha Sheshadri, Anhong Guo, and Kotaro Hara. Audio description customization. In Proceedings of the 26th International ACM SIGACCESS Conference on Computers and Accessibility, pages 1–19, 2024.

[21] OpenAI. Gpt-5.2 system card. Technical report, OpenAI, December 2025. Technical Report.

[22] Jaclyn Packer, Katie Vizenor, and Joshua A Miele. An overview of video description: history, benefits, and guidelines. Journal of Visual Impairment & Blindness, 109(2):83–93, 2015.

[23] Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings ofthe 40th annual meeting ofthe Association for Computational Linguistics, pages 311–318, 2002.

[24] Amy Pavel, Gabriel Reyes, and Jeffrey P. Bigham. Rescribe: Authoring and automatically editing audio descriptions. In Proceedings of the 33rd Annual ACM Symposium on User Interface Software and Technology (Virtual Event, USA) (UIST ’20), pages 747–759, New York, NY, USA, 2020. Association for Computing Machinery.

[25] Qwen Team. Qwen3.5-Omni technical report. arXiv preprint arXiv:2604.15804, 2026.

[26] A. Remael, N. Reviers, and G. Vercauteren. Pictures painted in words: Adlab audio description guidelines. https://dcmp.org/learn/captioningkey/624.

[27] Anna Rohrbach, Atousa Torabi, Marcus Rohrbach, Niket Tandon, Christopher Pal, Hugo Larochelle, Aaron Courville, and Bernt Schiele. Movie description. International Journal of Computer Vision, 123(1):94–120, 2017.

[28] SKERI. Youdescribe. the smith-kettlewell eye research institute. https://youdescribe. org/, 2022.

[29] Mattia Soldan, Alejandro Pardo, Juan León Alcázar, Fabian Caba, Chen Zhao, Silvio Giancola, and Bernard Ghanem. Mad: A scalable dataset for language grounding in videos from movie audio descriptions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5026–5035, 2022.

[30] W3C. G8: Providing a movie with extended audio descriptions. https://www.w3.org/TR/ WCAG20-TECHS/G8.html.

[31] W3C. Audio description (prerecorded): Understanding sc 1.2.5. https://www.w3.org/TR/ UNDERSTANDING-WCAG20/media-equivaudio-desc-only.html, 2022.

[32] Yujia Wang, Wei Liang, Haikun Huang, Yongqi Zhang, Dingzeyu Li, and Lap-Fai Yu. Toward automatic audio description generation for accessible videos. In Proceedings ofthe 2021 CHI Conference on Human Factors in Computing Systems, pages 1–12, New York, NY, USA, 2021. Association for Computing Machinery.

[33] Tsung-Han Wu, Joseph E Gonzalez, Trevor Darrell, and David M Chan. Claira: Leveraging large language models to judge audio captions. In arXiv preprint arXiv:2409.12962, 2024.

[34] Junyu Xie, Tengda Han, Max Bain, Arsha Nagrani, Gül Varol, Weidi Xie, and Andrew Zisserman. Autoad-zero: A training-free framework for zero-shot audio description. In Proceedings of the Asian Conference on Computer Vision, pages 2265–2281, 2024.

[35] Zihao Yue, Qi Zhang, Anwen Hu, Liang Zhang, Ziheng Wang, and Qin Jin. Movie101: A new movie understanding benchmark. arXiv preprint arXiv:2305.12140, 2023.

[36] Beste F Yuksel, Pooyan Fazli, Umang Mathur, Vaishali Bisht, Soo Jung Kim, Joshua Junhee Lee, Seung Jung Jin, Yue-Ting Siu, Joshua A Miele, and Ilmi Yoon. Human-in-the-loop machine learning to increase video accessibility for visually impaired and blind users. In Proceedings of

the 2020 ACM Designing Interactive Systems Conference, pages 47–60, New York, NY, USA, 2020. Association for Computing Machinery.

[37] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q Weinberger, and Yoav Artzi. Bertscore: Evaluating text generation with bert. arXiv preprint arXiv:1904.09675, 2019.

## Appendix

## A Synthesized Suite of metrics from Expert Guidelines

This section presents the full suite of evaluation metrics we synthesized based on expert guidelines. AD Script Content Quality:

• G1 - Content Coverage: AD describes important visual content necessary to understand the video.

– G1.1 - Prioritization: AD prioritizes visual information necessary to understand the video.

– G1.2 - Redundancy: AD does not describe information that is already clear to BLV viewers from the audio alone (e.g., redundant with speech or sounds).

– G1.3 - On-Screen Text: AD narrates on-screen text.

– G1.4 - Visual Style: AD describes style, color, textures and temporal transitions only as useful to understand the video (audience members differ in their preferences for descriptions around visual styles [11, 17, 20]).

• G2 - Content Timeliness: AD describes visual content close to the time it appears on screen.

• G3 - Content Accuracy: AD contains only accurate information (similar metrics already addressed in prior work [9]).

• G4 - Content Objectivity: AD describes content as it appears on screen and does not include censorship or interpretation (audience members differ in their preferences for objective vs. subjective descriptions [11, 17]).

## AD Script Style:

• G5 - Active Voice: AD script is written in third person with active voice.

• G6 - Appropriateness: Language of the AD matches the language of the video such that it is appropriate for the audience.

– G6.1 - Consistency: AD uses language that is consistent within the AD and with the speech in the video.

– G6.2 - Terminology and Tone: AD uses vocabulary and tone that are similar to the vocabulary and tone in the video (e.g., does not introduce jargon, uses similar level of formality).

## AD Audio Quality:

• G7 - Overlaps: AD avoids overlapping speech.

• G8 - Audio Coverage: AD covers the majority of the video, but does not need to fill every silence.

• G9 - Pronunciation: AD pronounces words correctly.

• G10 - Appropriate Speech Style: Speed and prosody are appropriate for the content of the video (e.g., somber for a melancholy scene).

For the analysis of the main experiments in this work, we use a subset of our guidelines that we prioritized for evaluation based on a pilot study with BLV audience members reviewing descriptions created with a basic prompt (e.g., “Create an audio description”). In the pilot study BLV users surfaced issues that most negatively impacted their experience, and these issues informed our choice of guidelines for initial metrics (G1.1-1.3, G2, G7, G8).

## B Full Results on the Proposed Metrics

This section gives the full results behind Figs. 3 to 5: the prompt comparison (App. Tab. 4), the six initial models (App. Tab. 5), and the newer omni-modal models (App. Tab. 6). The prompt

comparison uses the validation videos for which every prompt has output. The model tables score each model on all 101 test videos it has output for (Qwen-3.5-Omni-Flash: 57, Qwen-3.5-Omni-Plus: 99), so a model’s values agree across tables, but Gemini-2.5 Flash with the Gd. + Timing ASR prompt differs between the prompt table and the model tables. Audio Coverage is better when higher only up to 1; values above 1 mean the AD is longer than all available silence (Sec. 4). We therefore shade the highest values that $\mathrm { a r e } \leq 1$
<table><tr><td rowspan=1 colspan=6>Metric                       Simple AD Guideline only  Guideline + ASR Gd. + Timing ASR  Human AD</td></tr><tr><td rowspan=1 colspan=2>Prioritization, $\mathrm { G 1 . 1 } \left( \uparrow \right)$            $0 . 7 5 \pm 0 . 0 2$ </td><td rowspan=1 colspan=1> $0 . 7 9 \pm 0 . 0 2$ </td><td rowspan=1 colspan=1> $0 . 7 9 \pm 0 . 0 2$ </td><td rowspan=1 colspan=2> $0 . 7 6 \pm 0 . 0 2$         $0 . 7 3 \pm 0 . 0 2$ </td></tr><tr><td rowspan=1 colspan=1>No Redundancy, $\mathrm { { G 1 } } \dot { . } \dot { 2 } \left( \uparrow \right)$ </td><td rowspan=1 colspan=1> $0 . 7 7 \pm 0 . 0 3$ </td><td rowspan=1 colspan=1> $0 . 7 2 \pm 0 . 0 2$ </td><td rowspan=1 colspan=1> $0 . 7 1 \pm 0 . 0 2$ </td><td rowspan=1 colspan=1> $0 . 6 9 \pm 0 . 0 3$ </td><td rowspan=1 colspan=1> $0 . 7 2 \pm 0 . 0 2$ </td></tr><tr><td rowspan=1 colspan=1>On-Screen Text, $\mathrm { G } 1 . 3 \ : ( \uparrow )$ </td><td rowspan=1 colspan=1> $0 . 4 7 \pm 0 . 0 3$ </td><td rowspan=1 colspan=1> $0 . 6 9 \pm 0 . 0 3$ </td><td rowspan=1 colspan=1> $0 . 6 5 \pm 0 . 0 3$ </td><td rowspan=1 colspan=1> $0 . 6 3 \pm 0 . 0 3$ </td><td rowspan=1 colspan=1> $0 . 6 1 \pm 0 . 0 3$ </td></tr><tr><td rowspan=1 colspan=1>Timeliness, G2 (s) (↓)</td><td rowspan=1 colspan=1> $6 . 4 6 \pm 3 . 0 3$ </td><td rowspan=1 colspan=1> $3 . 2 7 \pm 1 . 2 9$ </td><td rowspan=1 colspan=1> $5 . 1 1 \pm 2 . 0 7$ </td><td rowspan=1 colspan=1> $4 . 4 3 \pm 1 . 1 7$ </td><td rowspan=1 colspan=1> $2 . 9 0 \pm 0 . 4 8$ </td></tr><tr><td rowspan=1 colspan=1>Audio Overlaps, $\mathrm { G } 7 \left( \mathrm { s } \right) \left( \downarrow \right)$ </td><td rowspan=1 colspan=1> $1 5 . 0 7 \pm 3 . 0 1$ </td><td rowspan=1 colspan=1> $5 9 . 1 9 \pm 7 . 2 8$ </td><td rowspan=1 colspan=1> $2 1 . 3 4 \pm 4 . 6 4$ </td><td rowspan=1 colspan=1> $1 2 . 7 7 \pm 3 . 8 8$ </td><td rowspan=1 colspan=1> $7 . 5 1 \pm 1 . 4 2$ </td></tr><tr><td rowspan=1 colspan=1>Audio Coverage, $\mathrm { G 8 } \ \mathrm { ( r a t i o , \le 1 ) }$ </td><td rowspan=1 colspan=1> $0 . 4 1 \pm 0 . 0 6$ </td><td rowspan=1 colspan=1> $1 . 2 3 \pm 0 . 1 1$ </td><td rowspan=1 colspan=1> $0 . 8 3 \pm 0 . 0 8$ </td><td rowspan=1 colspan=1> $0 . 7 8 \pm 0 . 0 5$ </td><td rowspan=1 colspan=1> $0 . 6 1 \pm 0 . 0 3$ </td></tr><tr><td rowspan=1 colspan=2>Avg. Silence, G8 (s) (↓)           $1 7 . 9 5 \pm 3 . 4 6$ </td><td rowspan=1 colspan=1> $8 . 8 5 \pm 3 . 1 3$ </td><td rowspan=1 colspan=1> $6 . 7 1 \pm 2 . 4 7$ </td><td rowspan=1 colspan=1> $8 . 9 7 \pm 2 . 8 0$ </td><td rowspan=1 colspan=1> $4 . 1 9 \pm 1 . 1 3$ </td></tr><tr><td rowspan=1 colspan=1>VideoA11y-Ref Descriptive (↑)</td><td rowspan=1 colspan=1> $3 . 4 6 \pm 0 . 1 3$ </td><td rowspan=1 colspan=1> $3 . 2 8 \pm 0 . 1 4$ </td><td rowspan=1 colspan=2> $2 . 8 1 \pm 0 . 1 1$            $3 . 0 3 \pm 0 . 1 0$ </td><td rowspan=1 colspan=1> $4 . 7 2 \pm 0 . 0 7$ </td></tr><tr><td rowspan=1 colspan=1>VideoA11y-Ref Objective (↑)</td><td rowspan=1 colspan=1> $3 . 5 5 \pm 0 . 1 3$ </td><td rowspan=1 colspan=1> $3 . 5 9 \pm 0 . 1 3$ </td><td rowspan=1 colspan=1> $3 . 2 7 \pm 0 . 1 2$ </td><td rowspan=1 colspan=1> $2 . 9 8 \pm 0 . 1 3$ </td><td rowspan=1 colspan=1> $4 . 9 3 \pm 0 . 0 3$ </td></tr><tr><td rowspan=1 colspan=1>VideoA11y-Ref Accurate (↑)</td><td rowspan=1 colspan=1> $1 . 8 2 \pm 0 . 1 1$ </td><td rowspan=1 colspan=1> $2 . 1 8 \pm 0 . 1 2$ </td><td rowspan=1 colspan=1> $1 . 7 4 \pm 0 . 1 0$ </td><td rowspan=1 colspan=1> $1 . 6 4 \pm 0 . 0 9$ </td><td rowspan=1 colspan=1> $4 . 9 8 \pm 0 . 0 2$ </td></tr><tr><td rowspan=1 colspan=1>VideoA11y-Ref Clear (↑)</td><td rowspan=1 colspan=1> $2 . 5 9 \pm 0 . 1 3$ </td><td rowspan=1 colspan=1>2.85 ±0.14</td><td rowspan=1 colspan=2> $2 . 5 7 \pm 0 . 1 2$            $2 . 6 8 \pm 0 . 1 1$ </td><td rowspan=1 colspan=1> $4 . 9 6 \pm 0 . 0 3$ </td></tr></table>

Table 4: Prompt Comparisons. AD generated by Gemini-2.5 Flash with each prompt, and the professiona Human AD, on validation videos with output from every prompt. Showing the mean and the 95% confidence interval, ↑ (higher is better); ↓ (lower is better). Cells are shaded by rank: First and Second .

<table><tr><td>Metric</td><td>Gemini-2.5 Flash</td><td>Qwen-2-VL-Max</td><td>GPT-4.1 mini</td><td>Claude 4</td><td>Claude 4.5</td><td>GPT-5.2</td></tr><tr><td>Prioritization,  $\mathrm { G 1 . 1 \it ( \uparrow ) }$ </td><td>0.76 ±0.04</td><td> $0 . 5 9 \pm 0 . 0 5$ </td><td> $0 . 5 2 \pm 0 . 0 5$ </td><td> $0 . 5 1 \pm 0 . 0 5$ </td><td> $0 . 5 9 \pm 0 . 0 5$ </td><td> $0 . 5 5 \pm 0 . 0 5$ </td></tr><tr><td>No Redundancy,  $\mathrm { G } 1 . 2 ( \uparrow )$ </td><td>0.70 ±0.05</td><td>0.71 ±0.04</td><td> $0 . 7 0 \pm 0 . 0 5$ </td><td> $0 . 6 9 \pm 0 . 0 5$ </td><td> $0 . 7 0 \pm 0 . 0 4$ </td><td>0.73 ±0.04</td></tr><tr><td>On-Screen Text, G1.3 (↑)</td><td>0.64 ±0.06</td><td>0.59 ±0.06</td><td> $0 . 5 8 \pm 0 . 0 6$ </td><td> $0 . 5 6 \pm 0 . 0 5$ </td><td> $0 . 6 1 \pm 0 . 0 5$ </td><td>0.63 ±0.05</td></tr><tr><td>Timeliness, G2 (s) (↓)</td><td>5.00 ±3.07</td><td> $6 . 9 2 \pm 3 . 9 1$ </td><td> $6 5 . 2 8 \pm 1 8 . 5 9$ </td><td> $1 9 . 8 3 \pm { 4 . 3 9 }$ </td><td> $2 0 . 5 0 \pm 4 . 3 0$ </td><td> $2 2 . 5 6 \pm 4 . 7 6$ </td></tr><tr><td>Audio Overlaps,  $\mathrm { G } 7 \left( \mathrm { s } \right) \left( \downarrow \right)$ </td><td>8.92 ±5.62</td><td> $4 6 . 5 3 \pm 1 1 . 1 6$ </td><td> $2 1 . 4 6 \pm 4 . 7 7$ </td><td> $7 . 4 5 \pm 2 . 1 4$ </td><td> $1 2 . 8 6 \pm 3 . 3 1$ </td><td> $1 1 . 1 4 \pm 2 . 7 4$ </td></tr><tr><td>Audio Coverage, G8 (ratio, ≤1)</td><td> $0 . 7 4 \pm 0 . 0 7$ </td><td> $0 . 9 4 \pm 0 . 1 5$ </td><td> $1 . 4 8 \pm 0 . 4 6$ </td><td> $0 . 6 2 \pm 0 . 0 6$ </td><td> $0 . 7 0 \pm 0 . 0 7$ </td><td> $0 . 6 5 \pm 0 . 0 7$ </td></tr><tr><td>Avg. Silence, G8 (s) (↓)</td><td> $6 . 6 3 \pm 2 . 1 6$ </td><td> $8 . 8 1 \pm 2 . 7 8$ </td><td> $1 9 . 3 7 \pm 5 . 8 9$ </td><td> $1 8 . 5 7 \pm 8 . 7 8$ </td><td> $1 0 . 4 0 \pm 5 . 8 6$ </td><td> $1 4 . 8 5 \pm 7 . 3 1$ </td></tr><tr><td>VideoA11y-Ref Descriptive (↑)</td><td> $3 . 1 0 \pm 0 . 1 9$ </td><td> $2 . 5 8 \pm 0 . 2 2$ </td><td> $3 . 2 8 \pm 0 . 2 4$ </td><td> $2 . 3 2 \pm 0 . 1 8$ </td><td> $2 . 6 9 \pm 0 . 2 1$ </td><td> $2 . 8 9 \pm 0 . 2 1$ </td></tr><tr><td>VideoA11y-Ref Objective (↑)</td><td>2.92 ±0.22</td><td> $2 . 7 7 \pm 0 . 2 5$ </td><td> $2 . 8 2 \pm 0 . 2 2$ </td><td> $2 . 1 3 \pm 0 . 2 0$ </td><td> $2 . 7 7 \pm 0 . 2 4$ </td><td> $2 . 9 4 \pm 0 . 2 6$ </td></tr><tr><td>VideoA11y-Ref Accurate (↑)</td><td>1.63 ±0.15</td><td> $1 . 3 3 \pm 0 . 1 4$ </td><td> $1 . 3 1 \pm 0 . 1 4$ </td><td> $1 . 1 6 \pm 0 . 1 0$ </td><td> $1 . 2 6 \pm 0 . 1 3$ </td><td> $1 . 3 5 \pm 0 . 1 4$ </td></tr><tr><td>VideoA11y-Ref Clear (↑)</td><td>2.69 ±0.20</td><td>2.21 ±0.21</td><td> $2 . 1 9 \pm 0 . 2 3$ </td><td> $2 . 0 1 \pm 0 . 1 8$ </td><td> $2 . 2 1 \pm 0 . 2 2$ </td><td> $2 . 5 4 \pm 0 . 2 5$ </td></tr></table>

Table 5: Model Comparisons (Initial Frontier Models). Comparing frontier models under the $G d . \mathrm { ~ + ~ } T i m i n g$ ASR prompt on the 101 test videos. Showing the mean and the 95% confidence interval, ↑ (higher is better); ↓ (lower is better). Cells are shaded by rank: First and Second

<table><tr><td></td><td rowspan=1 colspan=7>Metric                   Gemini-2.5 Flash Gemini-3.5 Flash† Qwen-2-VL-Max Qwen-3.5-Omni-Flash† Qwen-3.5-Omni-Plus†</td></tr><tr><td></td><td rowspan=1 colspan=2>Prioritization, ${ \mathrm { G 1 . 1 } } \left( \uparrow \right)$             $0 . 7 6 \pm 0 . 0 4$ </td><td rowspan=1 colspan=1> $0 . 9 3 \pm 0 . 0 3$ </td><td rowspan=1 colspan=1> $0 . 5 9 \pm 0 . 0 5$ </td><td rowspan=1 colspan=1> $0 . 9 4 \pm 0 . 0 2$ </td><td rowspan=1 colspan=2> $0 . 7 9 \pm 0 . 0 6$ </td></tr><tr><td></td><td rowspan=1 colspan=1>No Redundanc $\mathsf { y } , \mathsf { G } 1 . 2 \left( \uparrow \right)$ </td><td rowspan=1 colspan=1> $0 . 7 0 \pm 0 . 0 5$ </td><td rowspan=1 colspan=1> $0 . 9 0 \pm 0 . 0 3$ </td><td rowspan=1 colspan=1> $0 . 7 1 \pm 0 . 0 4$ </td><td rowspan=1 colspan=1> $0 . 8 8 \pm 0 . 0 5$ </td><td rowspan=1 colspan=1>0.88 ±</td><td rowspan=1 colspan=1>0.04</td></tr><tr><td></td><td rowspan=1 colspan=1>On-Screen Text, $\mathbf { G } 1 . 3 \mathbf { \Gamma } ( \uparrow )$ </td><td rowspan=1 colspan=1> $0 . 6 4 \pm 0 . 0 6$ </td><td rowspan=1 colspan=1> $0 . 5 7 \pm 0 . 0 5$ </td><td rowspan=1 colspan=1> $0 . 5 9 \pm 0 . 0 6$ </td><td rowspan=1 colspan=1> $0 . 6 1 \pm 0 . 0 6$ </td><td rowspan=1 colspan=1> $0 . 5 8 \pm</td><td rowspan=1 colspan=1>0 . 0 6$ </td></tr><tr><td rowspan=4 colspan=3>Timeliness,Audio Overlaps, $\mathrm { G } 7 \left( \mathrm { s } \right) \left( \downarrow \right)$ Audio Coverage, $\mathbf { G } 8 \ ( \mathrm { r a t i o } , \leq 1 )$ Avg. Silence, $\mathrm { G 8 \left( s \right) ( \downarrow ) }$            $6 . 6 3 \pm 2 . 1 6$ </td><td rowspan=1 colspan=1>Timeliness, G2 (s) (↓)</td><td rowspan=1 colspan=1>5.00±3.07</td><td rowspan=1 colspan=1> $2 . 9 9 \pm 0 . 9 1$ </td><td rowspan=1 colspan=2> $6 . 9 2 \pm 3 . 9 1$ </td></tr><tr><td rowspan=1 colspan=1>8.92 ±5.62</td><td rowspan=1 colspan=1> $0 . 2 6 \pm 0 . 1 3$ </td><td rowspan=1 colspan=1> $4 6 . 5 3 \pm 1 1 . 1 6$ </td><td rowspan=1 colspan=1> $5 2 . 1 4 \pm 1 5 . 8 9$ </td><td rowspan=1 colspan=1> $3 0 . 4 5 \p</td><td rowspan=1 colspan=1>m 8 . 6 9$ </td></tr><tr><td rowspan=1 colspan=1>0.74 ±0.07</td><td rowspan=1 colspan=1> $0 . 4 5 \pm 0 . 0 3$ </td><td rowspan=1 colspan=1> $0 . 9 4 \pm 0 . 1 5$ </td><td rowspan=1 colspan=1> $0 . 9 4 \pm 0 . 2 0$ </td><td rowspan=1 colspan=2> $0 . 7 2 \pm 0 . 1 4$ </td></tr><tr><td rowspan=1 colspan=1> $4 . 8 5 \pm 1 . 2 0$ </td><td rowspan=1 colspan=1> $8 . 8 1 \pm 2 . 7 8$ </td><td rowspan=1 colspan=1> $1 6 . 6 1 \pm 1 1 . 8 8$ </td><td rowspan=1 colspan=2> $9 . 4 9 \pm 2 . 1 2$ </td></tr><tr><td rowspan=4 colspan=3> $\mathrm { \ V i d e o A 1 1 y { - } R e f D e s c r i p t i v e \ ( \uparrow ) }$       $3 . 1 0 \pm 0 . 1 9$ VideoA11y-Ref Objective (↑)       $2 . 9 2 \pm 0 . 2 2$ VideoA11y-Ref Accurate (↑)        $1 . 6 3 \pm 0 . 1 5$ VideoA11y-Ref Clear (↑)          $2 . 6 9 \pm 0 . 2 0$ </td><td rowspan=1 colspan=1> $4 . 0 2 \pm 0 . 1 5$ </td><td rowspan=1 colspan=1> $2 . 5 8 \pm 0 . 2 2$ </td><td rowspan=1 colspan=1> $4 . 4 3 \pm 0 . 2 3$ </td><td rowspan=1 colspan=2> $3 . 8 2 \pm 0 . 2 7$ </td></tr><tr><td rowspan=1 colspan=1> $4 . 0 7 \pm 0 . 1 5$ </td><td rowspan=1 colspan=1> $2 . 7 7 \pm 0 . 2 5$ </td><td rowspan=1 colspan=1> $4 . 4 1 \pm 0 . 2 0$ </td><td rowspan=1 colspan=2> $4 . 0 3 \pm 0 . 2 6$ </td></tr><tr><td rowspan=1 colspan=1> $3 . 7 9 \pm 0 . 1 8$ </td><td rowspan=1 colspan=1> $1 . 3 3 \pm 0 . 1 4$ </td><td rowspan=1 colspan=1> $3 . 9 8 \pm 0 . 2 5$ </td><td rowspan=1 colspan=2> $3 . 2 3 \pm 0 . 2 5$ </td></tr><tr><td rowspan=1 colspan=1> $4 . 0 2 \pm 0 . 1 7$ </td><td rowspan=1 colspan=1> $2 . 2 1 \pm 0 . 2 1$ </td><td rowspan=1 colspan=1> $4 . 2 7 \pm 0 . 2 6$ </td><td rowspan=1 colspan=2> $3 . 8 1 \pm 0 . 2 7$ </td></tr></table>

Table 6: Next-Generation Omni-Modal Model Comparisons. Comparing Gemini-2.5/3.5 Flash and $\scriptstyle { \mathrm { Q w e n } } - 2 .$ VL-Max vs. Qwen-3.5-Omni-Flash/Plus under the Gd. + Timing ASR prompt. Each model is scored on all test videos it has output for: 101 for Gemini-2.5/3.5 Flash and Qwen-2-VL-Max, 99 for Qwen-3.5-Omni-Plus, and 57 for Qwen-3.5-Omni-Flash. Showing the mean and 95% confidence interval, ↑ (higher is better); ↓ (lower is better). For models marked with <sup>†</sup>, all LLM-judged metrics (G1.1, G1.2, G2, and VideoA11y-Ref) used Gemini-3.5 Flash as the judge; the other models used Gemini-2.5 Flash. The segment-level metrics (G1.1, G1.2, G2) appear robust to this change, whereas the holistic VideoA11y-Ref ratings are more sensitive to it; programmatic metrics (G1.3, G7, G8) do not depend on the judge. Cells are shaded by rank: First and Second .

## C Inter Annotator Agreement and Sighted Rater Prompt Comparisons

This section presents the detailed inter-annotator correlation and scoring statistics for our Sighted Rater study (App. Tab. 7) alongside the sighted rater evaluation across prompting strategies (App. Fig. 10).

<table><tr><td>Metric</td><td>Krippendorff&#x27;s α</td><td>Avg Std Dev</td></tr><tr><td>Timeliness</td><td>0.721</td><td>0.581</td></tr><tr><td>Audio Issues</td><td>0.700</td><td>0.545</td></tr><tr><td>Objective</td><td>0.665</td><td>0.437</td></tr><tr><td>Gaps of Silence</td><td>0.647</td><td>0.637</td></tr><tr><td>Overall Quality</td><td>0.581</td><td>0.681</td></tr><tr><td>Descriptive</td><td>0.579</td><td>0.699</td></tr><tr><td>Relevance</td><td>0.563</td><td>0.641</td></tr><tr><td>Accuracy</td><td>0.547</td><td>0.706</td></tr><tr><td>Clarity</td><td>0.544</td><td>0.709</td></tr><tr><td>Coverage</td><td>0.539</td><td>0.700</td></tr><tr><td>Consistency/ Redundancy</td><td>0.537</td><td>0.696</td></tr></table>

Table 7: Inter-Annotator Agreement (Krippendorff’s α) and Average Standard Deviation per Metric. Overall (all metrics): 0.654

![](images/d5580f80528edd58313d089001f3c6ad14beafc9f5a57103c4c0ca37c834c9a7.jpg)  
Figure 10: Sighted rater prompt comparisons. Mean Likert scores across five (of 11) key dimensions (↑ indicates higher is better). Sighted raters evaluated videos played with AD generated from different prompts.

## D Full Distribution of Scores and Qualitative Insights from the BLV User Study

This section visually documents the rating patterns observed across the blind and low-vision cohort evaluation (App. Figs. 11 and 12) and summarizes qualitative participant feedback.

## D.1 BLV Qualitative Feedback and Individual Preference Variance

Consistent with prior accessibility literature [11, 17, 20], post-task interviews with our 7 BLV participants (28 paired video evaluations) revealed substantial individual-level and genre-dependent variance in AD preferences. While professional human AD was favored overall (53.6%) for its smooth narrative flow, concise phrasing, and natural integration around dialogue pauses, AutoAD (Gemini-2.5 Flash with Gd. + Timing ASR) was preferred in 46.4% of comparisons—particularly on visually dense travel, nature, and instructional videos where participants valued exhaustive enumeration of visual objects, settings, and on-screen text. Conversely, when AutoAD was penalized, participants cited occasional redundancy with sound effects or abrupt sentence truncation when an AD segment reached the onset of video speech.

![](images/5642728bcba5ed99e4d25f02a2259e13445c7333b675de937cf08434f55ea8e2.jpg)  
Figure 11: BLV study. Distribution of Likert scores across 8 key dimensions. BLV participants evaluated videos played with human written ADs and AutoAD (Gemini-2.5 Flash with Gd. + Timing ASR). They rated both AD versions consistently high (ratings of 4 or 5) across all metrics. A few of the videos with automated AD was rated lower with respect to audio quality, and in terms of overall quality and satisfaction when compared to the same videos played with human generated AD.

![](images/fb601d0ee3a59dcf872f2e5106fbe296c0bde42553c76ed3833aa28a55be620d.jpg)  
Figure 12: BLV Study. In a blind study, BLV participants listening to the same video with human generated AD and AutoAD (Gemini-2.5 Flash with Gd. + Timing ASR) when asked to pick which AD version they liked better, marginally preferred the video version with human generated AD.

## E Extended Empirical Metric Validation and Reference-Free Evaluation

This section provides the full empirical correlation tables between our automated evaluation metrics and human judgments across the Sighted Rater cohort $( n = 4 2$ videos, App. Tab. 8), the BLV Participant cohort $( n = 2 8$ video evaluations, App. Tab. 9), and our Reference-Free Quality-RF comparison (App. Tab. 10).

Table 8: Automated Metrics vs. Sighted Rater Judgments (n = 42 videos). Video-level Spearman rank correlations (ρ) and p-values alongside qualitative interpretations.
<table><tr><td>Automated Metric</td><td>Metric Type</td><td>Target Human Dimension</td><td>Spearman ρ</td><td>p-value</td><td>Interpretation</td></tr><tr><td>Audio Overlaps (G7)</td><td>Programmatic (TTS + ASR)</td><td>Audio Issues (No Overlaps)</td><td>-0.467</td><td>0.0018**</td><td>Moderate alignment (fewer overlap seconds → higher human audio rating)</td></tr><tr><td>Avg. Silence Duration (G8)</td><td>Programmatic (TTS + ASR)</td><td>Gaps of Silence (Few Quiet Gaps)</td><td>-0.415</td><td>0.0063**</td><td>Moderate alignment (shorter silent gaps →</td></tr><tr><td>VideoA11y-Ref Clear</td><td>LLM Judge</td><td>Clarity</td><td>+0.453</td><td>0.0026**</td><td>higher human rating) Moderate positive alignment</td></tr><tr><td>VideoA11y-Ref Descriptive</td><td>LLM Judge</td><td>Descriptive</td><td>+0.328</td><td>0.0338*</td><td>Statistically significant positive alignment</td></tr><tr><td>Prioritization (G1.1)</td><td>LLM Judge (Visual Info)</td><td>Coverage</td><td>+0.246</td><td>0.1160</td><td>Positive trend</td></tr><tr><td>Timeliness (G2)</td><td>LLM Judge (Visual Info)</td><td>Timeliness</td><td>-0.228</td><td>0.1470</td><td>Expected negative trend (lower offset seconds → higher human rating)</td></tr></table>

Ablation of Reference-Free Input Representations. We evaluated four reference-free input configurations for Quality-RF: (1) video-only, (2) video + ASR transcript, (3) dense visual $i n f o + A S R .$ and (4) video + dense visual $i n f o + A S R .$ We found that providing the raw video + ASR transcript (App. Tab. 10, App. Fig. 26) achieves the strongest overall correlation across all four dimensions $( \rho = + 0 . 5 8 3$ on Descriptive, $\rho = + 0 . 4 1 5$ on Clear, and $\rho = + 0 . 2 4 4$ on Accurate), whereas text-only dense visual info + ASR achieves the highest standalone Accuracy correlation $( \rho = + 0 . 2 8 9 )$ when raw video frames cannot be passed to the judge.

Table 9: Automated Metrics vs. BLV Listener Judgments (n = 28 video evaluations). Video-level Spearman rank correlations (ρ) and p-values.
<table><tr><td>Automated Metric</td><td>Metric Type</td><td>Target BLV Dimension</td><td>Spearman ρ</td><td>p-value</td><td>Interpretation / Note</td></tr><tr><td>VideoA11y-Ref Clear</td><td>LLM Judge (Ref-based)</td><td>Clarity</td><td>+0.388</td><td>0.0411*</td><td>Statistically significant positive alignment</td></tr><tr><td>No Redundancy (G1.2)</td><td>LLM Judge (Ref-free)</td><td>Consistency / Non-Redundancy</td><td>+0.386</td><td>0.0427*</td><td>Statistically significant positive alignment</td></tr><tr><td>VideoA11y-Ref Descriptive</td><td>LLM Judge (Ref-based)</td><td>Descriptive</td><td>+0.340</td><td>0.0771</td><td>Marginal positive alignment (p &lt; 0.10)</td></tr><tr><td>VideoA11y-Ref Objective</td><td>LLM Judge (Ref-based)</td><td>Objective</td><td>+0.335</td><td>0.0811</td><td>Marginal positive alignment (p &lt; 0.10)</td></tr><tr><td>Audio Overlaps (G7)</td><td>Programmatic (Ref-free)</td><td>Audio Quality</td><td>+0.132</td><td>0.5024</td><td>Low correlation because the study player trun- cated AD at speech onset</td></tr><tr><td>VideoA11y-Ref Accurate</td><td>LLM Judge (Ref-based)</td><td>Perceived Accuracy</td><td>-0.264</td><td>0.1743</td><td>Negative correlation due to reference-matching bias (resolved by Quality-RF)</td></tr></table>

Table 10: Reference-Free (Quality-RF) vs. Reference-Based (VideoA11y-Ref) Correlation with BLV Human Ratings (n = 28). Replacing the human reference text with raw video and ASR transcript reverses the negative accuracy correlation and strengthens alignment across all four rubric dimensions.
<table><tr><td>VideoA11y Dimension</td><td>Ref-Based ρ</td><td>Ref-Based p</td><td>Ref-Free ρ (Video + ASR)</td><td>Ref-Free p</td><td>Key Finding</td></tr><tr><td>Descriptive</td><td>+0.340</td><td>0.0771</td><td>+0.583</td><td>0.0011**</td><td>+0.243 ρ gain; highly significant</td></tr><tr><td>Clear</td><td>+0.388</td><td>0.0411*</td><td>+0.415</td><td>0.0281*</td><td>+0.027ρ gain; statistically significant</td></tr><tr><td>Objective</td><td>+0.335</td><td>0.0811</td><td>+0.365</td><td>0.0561</td><td>+0.030 ρ gain; strong marginal trend</td></tr><tr><td>Accurate</td><td>-0.264</td><td>0.1743</td><td>+0.244</td><td>0.2109</td><td>Reverses negative correlation (+0.508 swing)</td></tr></table>

TTS Speaking Rate. Our temporal metrics (G7, G8) render AD segments with an off-the-shelf TTS engine at a fixed 1.2× speaking rate. Because a segment’s spoken duration scales inversely with the rate, faster speech shortens AD intervals and reduces overlap seconds, and slower speech lengthens them; the absolute values of these metrics therefore depend on the chosen rate. We do not study other rates here. Faster or adaptive per-gap speaking rates are a promising direction, as they would allow more descriptions to fit into the available silences.

## F Questions for Sighted Raters

This section documents the specific inquiry components provided to non-visually impaired participants. It lists each question and its corresponding categorical option to detail our human evaluation mechanics.

1. Coverage: The descriptions covered the important visual information necessary for a blind user to understand the video.

□ Strongly disagree. All important information was not covered.

□ Slightly disagree. Most important information was not covered.

□ Neither agree or disagree. Some important information was covered and some important information was missing.

□ Slightly agree. Most important information was covered.

□ Strongly agree. All important information was covered.

2. Accuracy: The descriptions were accurate.

□ Strongly disagree. All of the information was inaccurate (potentially misleading).

□ Slightly disagree. Most information was inaccurate.

□ Neither agree or disagree. Some information was accurate and some was inaccurate.

□ Slightly agree. Most information was accurate.

□ Strongly agree. All information was accurate.

3. Relevance: The descriptions included relevant information.

□ Strongly disagree. All of the information was irrelevant (not needed for a blind user to follow the video).

□ Slightly disagree. Most information was irrelevant.

□ Neither agree or disagree. Some information was relevant and some was irrelevant.

□ Slightly agree. Most information was relevant.

□ Strongly agree. All information was relevant.

4. Consistency/Redundancy: The descriptions account for information user already has from the video’s audio and prior descriptions.

□ Strongly disagree. All descriptions repeated information the user would already know from the video’s audio or prior descriptions.

□ Slightly disagree. Most descriptions repeated information the user would already know from the video’s audio or prior descriptions.

□ Neither agree or disagree. Some of the descriptions provided new information but some of them included information the user would already know from the audio or prior descriptions.

□ Slightly agree. Most descriptions provided new information but a few descriptions included information the user would already know from the video’s audio or prior descriptions.

□ Strongly agree. All descriptions provided new information that supplemented the existing video’s audio and prior descriptions.

5. Overall Quality: Overall, the content of the descriptions was high quality and engaging to listen to without the visuals.

□ Strongly disagree. All of the content of the narration had significant quality issues such that it was difficult to follow or not engaging to listen to. For example, the narration was overly repetitive and dull, or it had inconsistent references to characters, objects, or settings that made it difficult to follow.

□ Slightly disagree. Most of the content of the narration had significant quality issues such that it was difficult to follow or not engaging to listen to.

□ Neither agree or disagree. Some of the content of the narration was easy to follow and engaging to listen to. Some moments had quality issues that made the narration difficult to follow or not engaging to listen to.

□ Slightly agree. Most of the content of the narration was easy to follow and engaging to listen to. The language of the narration was mostly clear, illustrative, and engaging.

□ Strongly agree. All of the content of the narration was easy to follow and engaging to listen to. The language of the narration was entirely clear, illustrative, and engaging.

## 6. Clarity: The descriptions were clear and easy to follow considering the audio alone.

□ Strongly disagree. All descriptions were unclear or hard to follow when considering the information in the audio alone (e.g., used unclear or inconsistent references to characters, settings, or objects).

□ Slightly disagree. Most descriptions were unclear or hard to follow, but some descriptions were easy to follow when considering the information in the audio alone.

□ Neither agree or disagree. Some of the descriptions were clear and easy to follow, and some were not clear or hard to follow when considering the information in the audio alone.

□ Slightly agree. Most of the descriptions were clear and easy to follow when considering the information in the audio alone.

□ Strongly agree. All descriptions were clear and easy to follow from the audio alone.

## 7. Descriptive: The descriptions were descriptive in that they provided vivid details about objects, people, and settings while maintaining a concise narrative flow.

□ Strongly disagree. All of the descriptions were not descriptive. Instead, they were dull or repetitive, providing little useful information.

□ Slightly disagree. Most of the descriptions were not descriptive, but instead they were dull or repetitive.

□ Neither agree or disagree. Some of the descriptions included vivid details and some were dull or repetitive.

□ Slightly agree. Most of the descriptions included vivid details and the description mostly maintained a concise narrative flow.

□ Strongly agree. All of the descriptions included vivid details while maintaining a concise narrative flow.

## 8. Objective: The descriptions were objective in that they report what is visible without adding personal opinions or assumptions (e.g., about character’s relationships, or identity attributes).

□ Strongly disagree. All of the descriptions contained subjective opinions or assumptions.

□ Slightly disagree. Most of the descriptions included subjective opinions or assumptions.

□ Neither agree or disagree. Some of the descriptions were objective but some of the descriptions included subjective opinions or assumptions.

□ Slightly agree. Most of the descriptions were objective but some were subjective.

□ Strongly agree. All of the descriptions were objective.

## 9. Timeliness: The descriptions appeared close to the time the event occurred in the video (usually within 3-5s).

□ Strongly disagree. All events were not described close to the time they occurred.

□ Slightly disagree. Most events were not described close to the time they occurred.

□ Neither agree or disagree. Some events were described close to the time they occurred.

□ Slightly agree. Most events were described close to the time they occurred.

□ Strongly agree. All events were described close to the time they occurred.

## 10. Audio Issues: All descriptions were clear to listen to and did not have audio issues such as overlapping speech.

□ Strongly disagree. All descriptions had audio issues such as overlapping speech. It is difficult to listen to the descriptions due to audio issues.

□ Slightly disagree. Most descriptions had audio issues such as overlapping speech.

□ Neither agree or disagree. There were some descriptions that were clear and some that would be difficult to listen to due to audio issues.

□ Slightly agree. Most descriptions were clear without audio issues.

□ Strongly agree. All descriptions were clear without audio issues.

## 11. Gaps of Silence: There were not large gaps of silence in the descriptions (6-10s quiet gaps).

□ Strongly disagree. All gaps of silence in the video did not have descriptions.

□ Slightly disagree. There were frequently long gaps of silence throughout the video.

□ Neither agree or disagree. There were some long gaps of silence throughout the video.

□ Slightly agree. There were long gaps of silence once or twice in the video.

□ Strongly agree. There were no long gaps of silence.

## G Examples showing Inline Audio Description Constraints

![](images/3e031dadb0ddc5308779bf0106bc129484f2d62f5ad113d3c3468f7632882c83.jpg)  
Figure 13: Example demonstrating more detailed descriptions from human annotators relative to model generated description despite the tight timing constraints of the video.

![](images/e404c6b3558c778b64602bc64a8ea6ef51fcab55f0ba9f5f54178b69b1576142.jpg)  
Figure 14: Example demonstrating human annotators describing the perspective of the video in addition to vivid descriptions going with the flow of the narration.

## H Examples with different prompts

This section showcases descriptive side-by-side comparisons taken from standard frames in our repository. It features illustrative contrasts between automatically produced scripts and those developed by our expert pairs.

![](images/8759b4784ed229d2e97a1403dc7c5396f3834cc6feb0eda9b007cc18382f5577.jpg)  
Figure 15: Dataset Examples. Model Generated AD (created with the Simple AD and Gd. + Timing ASR prompts) and corresponding professionally annotated AD for a video segment. Here, we illustrate each video segment with manually selected video frames. Top example demonstrates narrative flow in the professionally annotated AD (describes woman’s story rather than just the actions), and the example below shows a missed event in the Model Generated ADs (the truck rapidly breaks for the elephant).

## I Prompts for AD Generation

This section displays the actual systemic directives fed into multimodal architectures to elicit narrations. It archives variations of baseline instructions, transcript integrations, and rigorous timing constraints.

Visual Information Prompt   
Please watch this video and list out all the important visual events that   
happen in the video narrative with timestamps. Please be very descriptive   
with time stamps.   
The output format must be exactly like this:   
\* One scene description per line.   
\* Each description follows the format <timestamp> <description>   
• \* The timestamp must be in the format 00:00:00.000 (hh:mm:ss.ms)   
followed by a "-".   
\* The descriptions must be sorted by timestamp in ascending order.   
\* \*\*CRUCIAL\*\*: The descriptions must follow this format and cover the   
entire video.   
Example output:   
00:00:00.000 visual\_event\_description1   
00:00:07.123 visual\_event\_description2   
00:00:23.745 visual\_event\_description3   
Afterwards please watch the video and list all the text on screen and when   
it appeared on the screen.   
Please list all people in the video if there are any and what we know about   
the people.   
Finally, please provide a summary of the video narrative.

Figure 16: Prompt used to generate detailed visual information in the video. The responses are used when evaluating the VLM and human generated descriptions.

Simple Prompt   
You are a professional audio description expert. Create inline audio   
descriptions for the provided video to describe visual events only when there   
is no speech or dialog. For each audio description, provide the start time   
as a timestamp (e.g., 01:12:45.403 with HH:MM:SS.MS) then provide the text of   
the description.  
Figure 17: Most basic baseline prompt used for generating audio description.

Guideline AD Prompt   
You are an expert audio description writer.   
Your task is to write high-quality audio descriptions for videos, ensuring   
accessibility for visually impaired viewers. Audio descriptions should   
narrate key visual elements that are not already conveyed through the   
dialogue or other audio elements.   
Please adhere to the following guidelines:   
What to Describe:   
\* Describe what you see. Focus on the essential visual information needed   
to understand the scene.   
\* Be concise.   
\* Always read on-screen text exactly as it appears.   
\* Be factual, objective, and avoid speculation.   
\* Use proper terminology and names whenever possible.   
\* All of the descriptions will need to fit in the space between dialog   
(assume 1-3 words per second when spoken aloud). If there is no   
space, then you need to skip the description.   
\* Try to match the mood of the video with your descriptions.   
\* Describe color only when it is vital to the comprehension of content.   
\*\* CRUCIAL: \*\* Your descriptions must cover the entire video.

What NOT to Describe:   
\* Do not talk over the dialog or any other essential audio.   
\* Do not describe what can be inferred from the audio.   
\* Do not over-describe; less is more. Keep descriptions brief and to the   
point.   
\* Do not interpret or editorialize. Stick to describing what is visually   
present.   
\* Do not give away secrets, surprises, or sight gags before they happen.   
\* Do not censor (sex, violence, gore, emotions), describe them accurately.   
\* Do not describe obvious sound cues such as dialogue, a phone ringing, or a   
dog barking.   
Remember to only place descriptions between gaps of speech. Strive for   
clear, concise, and informative audio descriptions that enhance the viewing   
experience without distracting from the original content.   
The output format must be a valid JSON with a single field called "response"   
containing the following:   
\* One scene description per line.   
\* Each description follows the format <timestamp>-STANDARD <description>   
• \* The timestamp must be in seconds in the format 0:00:00.000   
(h:mm:ss.ms) followed by a "-"   
• \* The type of description should always be STANDARD   
• \* The description must be in a single line.   
\* The descriptions must be sorted by timestamp in ascending order.   
\* \*\*CRUCIAL\*\*: The descriptions must follow this format and cover the   
entire video.   
\* \*\*CRUCIAL\*\*: The format must be valid.

Figure 18: Prompt used to generate audio description with just the guidelines and no video transcripts.

Guideline AD and ASR Prompt   
You are an expert audio description writer.   
Your task is to write high-quality audio descriptions for videos, ensuring   
accessibility for visually impaired viewers. Audio descriptions should   
narrate key visual elements that are not already conveyed through the   
dialogue or other audio elements.   
The video transcript with timestamps (start and end time in seconds) is   
provided below. Make sure to generate descriptions only in between dialogs   
where there is no speech in the video. The descriptions should fit in the   
gaps when spoken out loud. (assume you can fit in 1-3 words per second)   
<asr\_transcript>   
ASR\_TRANSCRIPT\_GOES\_HERE\_IF\_AVAILABLE   
</asr\_transcript>   
Please adhere to the following guidelines:   
What to Describe:   
\* Describe what you see. Focus on the essential visual information needed   
to understand the scene.   
\* Be concise.   
\* Always read on-screen text exactly as it appears.   
\* Be factual, objective, and avoid speculation.   
\* Use proper terminology and names whenever possible.   
\* All of the descriptions will need to fit in the space between dialog   
(assume 1-3 words per second when spoken aloud). If there is no   
space, then you need to skip the description.   
\* Try to match the mood of the video with your descriptions.   
\* Describe color only when it is vital to the comprehension of content.   
\*\* CRUCIAL: \*\* Your descriptions must cover the entire video.   
What NOT to Describe:

![](images/a11e60c70b644e951d7312115c7ff5b649ae7f616a4ced9e07a2b783686a4903.jpg)  
Figure 19: Prompt used to generate audio description with the audio transcripts and the guidelines.

## Guideline + Timing ASR Prompt

You are an expert audio description writer.

Your task is to write high-quality audio descriptions for videos, ensuring accessibility for visually impaired viewers. Audio descriptions should narrate key visual elements that are not already conveyed through the dialogue or other audio elements.

The video transcript with timestamps (start and end time in seconds) is provided below. You MUST only generate descriptions in the silent gaps BETWEEN dialogue segments indicated by the timestamps.

ASR\_TRANSCRIPT\_GOES\_HERE\_IF\_AVAILABLE

## </asr\_transcript>

\*\*CRUCIAL TIMING & LENGTH RULES TO THINK AND FOLLOW:\*\*

1. \*\*No Overlap:\*\* Descriptions MUST NOT overlap with any spoken dialogue in the provided transcript. Use the timestamps to find silent gaps.

2. \*\*Calculate Gap Duration:\*\* For each gap, calculate the available time in seconds.

3. \*\*Strict Word Limit:\*\* The number of words in any description MUST NOT exceed the gap duration (in seconds) multiplied by 3. This means a maximum of 3 words per second of available gap time.

4. \*\*Sentence Limit:\*\* Each description associated with a single timestamp MUST NOT exceed \*\*three\*\* sentences.

5. \*\*Fit the Gap:\*\* The spoken description must comfortably fit within the calculated silent gap, respecting both word and sentence limits.

6. \*\*Skip if No Space:\*\* If a visual element cannot be described within the available time and sentence limits in any gap, you MUST skip describing it.

7. \*\*Silent videos:\*\* If the video is a silent video with no speech (just music) then add more descriptions to cover the entire video.

Please adhere to the following guidelines:

## What to Describe:

\* Describe what you see. Focus on the essential visual information needed to understand the scene.

\* Be concise and fit the description within the time and sentence constraints.

\* Always read on-screen text exactly as it appears.

\* Be factual, objective, and avoid speculation.

\* Use proper terminology and names whenever possible.

\* Try to match the mood of the video with your descriptions.

\* Describe color only when it is vital to the comprehension of content.

\*\* CRUCIAL: \*\* Your descriptions must cover the entire video, respecting the timing gaps.

## What NOT to Describe:

\* Do not talk over the dialog or any other essential audio.

\* Do not describe what can be inferred from the audio.

\* Do not over-describe; less is more. Keep descriptions brief and to the point.

\* Do not interpret or editorialize. Stick to describing what is visually present.

\* Do not give away secrets, surprises, or sight gags before they happen.

\* Do not censor (sex, violence, gore, emotions), describe them accurately.

\* Do not describe obvious sound cues such as dialogue, a phone ringing, or a dog barking.

Remember to only place descriptions between gaps of speech. Strive for clear, concise, and informative audio descriptions that enhance the viewing experience without distracting from the original content.

The output format must be a valid JSON with a single field called "response" containing the following:

\* One scene description per line.

\* Each description follows the format <timestamp>-STANDARD <description>

• \* The timestamp must be in seconds in the format 0:00:00.000   
(h:mm:ss.ms) indicating the START time of the description within   
a silent gap, followed by a "-"   
• \* The type of description should always be STANDARD   
• \* The description must be in a single line, containing no more   
than three sentences.   
\* The descriptions must be sorted by timestamp in ascending order.   
\* \*\*CRUCIAL\*\*: The descriptions must follow this format and cover the   
entire video.   
\* \*\*CRUCIAL\*\*: The format must be valid.

Figure 20: Prompt used to generate audio description with the guidelines, the timestamped transcript, and the duration of each available gap and the time needed to speak (Gd. + Timing ASR).

## J Prompts for AD Evaluation Metrics

This section exhibits the specific prompts leveraged to operate LLM-as-a-judge scoring functions. It logs queries responsible for testing parameters like redundancy checks, content coverage, and narration accuracy.

Prioritization Prompt   
You are evaluating if audio descriptions (AD) are covering visual information   
necessary to understand the video. For each line in the first AD output,   
score if it is covered in the visual information (0 - not covered, 1 - fully   
covered).   
Then, output only an array of all the scores like [0,0.5,1.0,1.0,0]   
First AD Output:   
<ad\_output>   
Second visual information of the video:   
<dense\_visual\_info>  
Figure 21: Prompt used to evaluate the prioritization of important events in the audio description based on detailed visual information.

![](images/ef35886f1940e2c833785b356c9b203c2caeb7b66b8527fbc72c658fe993bcbf.jpg)  
Figure 22: Initial version of the On-Screen Text (G1.3) metric, which used an LLM judge to count unsaid on-screen words. All reported G1.3 results instead use a programmatic word-level recall (Sec. 4.1): the fraction of on-screen-text words listed in the dense visual information that appear anywhere in the AD or the ASR transcript; no LLM judge is involved.

![](images/cb665fd2325ff3afd41dec153ed66b3f5c06f84dfbb8d2c3ea15dce401e30e79.jpg)  
Figure 23: Prompt used to evaluate the redundancy of audio descriptions with the original transcript.

![](images/d3db21b92ecd5a770982c0ecf5d756b86024d4bb1db19d6b1c775f32d9781a4d.jpg)  
Figure 24: Prompt for evaluating the timeliness of audio descriptions by comparing them to pre-extracted visual information.

![](images/694832f011eb947f8ce048d8fc095c048232716f0e69e41b3321155701d56691.jpg)

![](images/897672187d5897fd39b8c1ee144b7da1842aa2627641803732a18e94c45e3d12.jpg)

Figure 25: Prompt used to evaluate the Reference-Based VideoA11y (VideoA11y-Ref) metrics.  
Reference-Free Quality-RF Prompt (Video + ASR)   
You are an expert evaluator with a deep understanding of audio description   
quality, particularly for making video content accessible to blind and low   
vision (BLV) individuals.   
Your role is to assess and rate audio descriptions based on specific   
metrics. Audio descriptions should describe important visual events that

```jsonl
are not already captured in the existing dialog or audio of the video. The
provided ASR transcript represents the spoken content in the video.
Task: I have a video, an ASR transcript of the video, and a
model-generated audio description. Additionally, I have four evaluation
metrics: Descriptive, Objective, Accurate, and Clear.
Please evaluate the model-generated audio description using the above
metrics by comparing it to the video and the ASR transcript.
Before providing the ratings, please provide a brief summary of the video
and your rationale for the ratings.
Metrics Definition:
1. Descriptive: A descriptive description should provide vivid details
about objects, people, and settings while maintaining a concise narrative
flow. It should include essential information about the appearance of
individuals, such as their clothing, facial expressions, and actions, and
visual properties of objects, such as color and shape. For example, "A
smiling woman, wearing a loose white dress, types on a laptop in a softly
lit room."
2. Objective: An objective description should report what is visible
without adding personal opinions or assumptions. For instance, describe two
people as "a woman and a man" without adding any relationship context unless
it is mentioned. It should also avoid guessing personal attributes like
racial or gender identities unless explicitly clear.
3. Accurate: An accurate description should aim for precision in
describing visible elements without imagination. It should ensure that all
details, such as colors and spatial arrangements, are reported correctly.
For instance, "Blue sky with white clouds" instead of "White sky with blue
clouds" if that is what appears. Additionally, it should avoid adding
unnecessary details that do not contribute to a deeper understanding of the
scene.
4. Clear: A clear description should present information in the videos
in a way that is easy to follow for blind and low vision individuals. It
should describe the object or character’s properties before the actions or
relationships with other objects or characters. For example, "A man wearing
sunglasses plays the piano." Pronouns should only be used when it is clear
who they refer to, and the description should include any on-screen text
if it is central to understanding. For instance, "A man in a black polo
shirt is speaking. He is in a studio setting with a red couch and a blue
background featuring the text ’Talk of the Town’".
Input:
ASR transcript:
<asr_transcript>
ASR_TRANSCRIPT_GOES_HERE_IF_AVAILABLE
</asr_transcript>
Model-generated audio description: <generated_desc>
GENERATED DESC GOES HERE IF AVAILABLE
</generated_desc>
Output Format: Return the result as a string format dictionary with the
following structure. Please output the fields in this exact order:
{"Summary": "Your brief summary of the video here", "Rationale": "Your
rationale for the ratings here", "Descriptive": [Rating out of 5],
"Objective": [Rating out of 5], "Accurate": [Rating out of 5], "Clear":
[Rating out of 5]}
```  
Figure 26: Prompt used to evaluate the reference-free Quality-RF metrics (VideoA11y rubric) directly against the raw video and ASR transcript without requiring human ground-truth AD.