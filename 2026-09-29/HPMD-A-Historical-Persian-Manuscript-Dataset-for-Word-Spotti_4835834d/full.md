# HPMD: A Historical Persian Manuscript Dataset for Word Spotting with Line-Level Annotation

Saeid Firouzi Daghigh and Majid Iranpour Mobarakeh

Abstract—Large collections of historical Persian manuscripts have been digitized, but searching them is still slow and mostly manual. Historians usually want to find where a specific name, date, event, or topic appears, which is a word spotting problem rather than a full transcription problem. Progress on this task is limited by two things. First, there is almost no public dataset of historical Persian handwriting; the only notable resource, OpenITI MAKHZAN, contains a relatively small Persian portion. Second, word spotting models usually need word-level bounding boxes, which are very expensive to annotate. In this paper we introduce a new dataset of 223 pages, 3,678 lines, 37,631 words, and 130,630 characters, collected from diverse historical Persian books of poetry and prose and annotated by 11 annotators at the region, line, and text level. We also propose a baseline that is trained only with line-level annotations but returns word-level locations. A fine-tuned line detector finds text lines, and a fine-tuned CRNN recognizer trained with CTC produces a frame-by-character posterior matrix for each line. Instead of decoding the most probable character at each frame, the query is scored directly against this matrix, so visually similar characters in Persian such as be and pe no longer cause hard failures. The frame alignment also gives the horizontal position of the word inside the line. On the test set, the fine-tuned line detector reaches an F1 of 0.892, and posterior-based search raises the word spotting F1 from 0.487 to 0.558 and recall from 0.340 to 0.548 compared with exact matching on the decoded text, with the decision threshold selected on a heldout validation set. A PHOC attribute-embedding baseline that additionally receives oracle word boundaries at test time reaches an F1 of 0.449, below the proposed method despite its stronger supervision. We also report a distributional analysis of the dataset, a taxonomy of retrieval errors, and a per-condition breakdown of performance. Dataset and codes are available at: https://github.com/saeed5959/persian handwritten dataset

Index Terms—Historical document analysis, Persian handwriting, word spotting, dataset.

## I. INTRODUCTION

P <sup>ERSIAN</sup> <sup>manuscripts</sup> <sup>form</sup> <sup>one</sup> <sup>of</sup> <sup>the</sup> <sup>richest</sup> <sup>written</sup> <sup>her-</sup>itages in the world. Libraries and archives have scanned itages in the world. Libraries and archives have scanned many of them, yet most of these images are not searchable. A historian who wants to know where a particular person, place, date, or event is mentioned still has to read page after page. Automatic tools could shorten this work from months to minutes, but only if they are reliable on the kind of material historians actually use.

A natural first idea is to run an optical character recognition (OCR) or handwritten text recognition (HTR) model and then search the resulting text. This works poorly on historical

Persian manuscripts for two reasons. The first is accuracy. Persian script is cursive, letters change shape by position, and many letters share the same base shape and differ only in the number or placement of dots, for example be/pe/te/se or jim/che/he/khe. In old manuscripts dots are often faded, misplaced, or missing. A standard recognizer outputs only the single most probable character at each position, so one wrong character is enough to make a search fail. The second reason is location. Most OCR pipelines return a string for each line, not the position of each word inside the page, while a historian needs to see exactly where the word is.

Word spotting solves the location problem, but most word spotting methods are trained with word-level bounding boxes [13], [15]–[17]. Drawing these boxes is tedious. For a collection of 200 pages one may need to draw around 2,000 line regions but roughly 20,000 word boxes. For low-resource scripts and historical material, this cost is often the real reason there are fewer datasets.

Public data for Persian handwriting is also scarce. Existing Persian datasets such as Sadri [1], Khayyam [2], Hoda [3], FHT [4], and the recent MPHD [9] were written by modern writers on collection forms. Historical datasets exist for Arabic [6] and Urdu [7], and OpenITI MAKHZAN [8] covers several Arabic-script languages, but its Persian portion is small.

In this work we address both the data problem and the annotation cost problem. We collected and annotated a dataset of historical Persian manuscripts that covers both poetry and prose, including pages from the Divan of Hafez, the Shahnameh, and Majma al-Bayan. The annotation is done only at the region and line level. We then show that a model trained with this cheaper supervision can still locate individual words. The key idea is to keep the full posterior matrix of a CTCbased recognizer instead of its decoded text, and to search this matrix directly.

The main contributions of this paper are:

1) A new dataset of historical Persian handwriting with 223 pages, 3,678 lines, 37,631 words, and 130,630 characters, drawn from diverse books of poetry and prose and annotated with text regions, line regions, and transcriptions, together with a distributional analysis of line and word lengths, character frequencies, script styles, genre, illumination, and degradation.

2) A word spotting approach that scores queries against the CTC posterior probability matrix instead of the decoded string. This makes the search tolerant to confusions between similar characters and increases the F1 score over exact matching.

3) A complete and reproducible baseline, including line detection, recognition, an explicit threshold-selection protocol, a comparison against a PHOC attributeembedding method with stronger supervision, and an error analysis with visual examples.

## II. RELATED WORKS

## A. Handwriting Datasets

Public benchmarks have shaped progress in handwriting recognition. For English, the IAM database [5] remains the most widely used resource for line-level recognition and word spotting. For Arabic-script languages the picture is less complete.

Several Persian handwriting datasets have been released. Hoda [3] contains a large number of isolated handwritten digits and is limited to digit recognition. The Sadri database [1] was collected from 500 writers and includes digits, letters, words, and free text, with writer metadata. Khayyam [2] focuses on unconstrained Persian words and contains about 44,000 words, 60,000 letters, and 6,000 digits from 400 writers. FHT [4] provides line-level Persian text, and MPHD [9] recently added a multi-purpose dataset with 500 writers, line-level transcriptions, isolated characters, and demographic information. All of these datasets were written by modern writers on prepared forms. They do not show the layout, ornaments, script styles, and degradation found in historical manuscripts.

Historical datasets in related scripts are also available. VML-HD [6] provides 680 pages from five historical Arabic books annotated at the sub-word level and was designed for word spotting and recognition. The Urdu Katib dataset [7] contains more than 13,000 text lines written by calligraphers in the Nastaliq style. OpenITI MAKHZAN [8] gathers about 1,500 page images of Arabic, Persian, Ottoman Turkish, and Urdu manuscripts and prints with line-level segmentation and transcription. It is the closest resource to ours, but only a limited part of it is Persian, and it is mainly intended for training transcription models. Table I compares these datasets with ours.

## B. Handwritten Text Recognition

Modern HTR systems usually combine a convolutional feature extractor with a recurrent sequence model and a Connectionist Temporal Classification (CTC) output layer. Wang and Hu [12] showed that adding gated recurrent connections inside the convolutional layers improves context modeling for text recognition. Transformer-based models are now common as well. HATFormer [11] adapts a transformer encoderdecoder to historical handwritten Arabic, and a recent study on TrOCR for medieval manuscripts [18] shows that finetuning choices strongly affect accuracy on small historical datasets. These systems output a text string, and their accuracy is usually reported with character or word error rates. For search, however, a single wrong character in the output is enough to miss a word.

TABLE I  
COMPARISON OF OUR DATASET WITH RELATED HANDWRITING DATASETS.
<table><tr><td>Dataset</td><td>Persian</td><td>Handwritten</td><td>Historical</td></tr><tr><td>VML-HD [6]</td><td>No</td><td>Yes</td><td>Yes</td></tr><tr><td>Urdu Katib [7]</td><td>No</td><td>Yes</td><td>Yes</td></tr><tr><td>IAM [5]</td><td>No</td><td>Yes</td><td>No</td></tr><tr><td>Sadri [1]</td><td>Yes</td><td>Yes</td><td>No</td></tr><tr><td>Khayyam [2]</td><td>Yes</td><td>Yes</td><td>No</td></tr><tr><td>Hoda [3]</td><td>Yes</td><td>Yes</td><td>No</td></tr><tr><td>FHT [4]</td><td>Yes</td><td>Yes</td><td>No</td></tr><tr><td>MPHD [9]</td><td>Yes</td><td>Yes</td><td>No</td></tr><tr><td>OpenITI MAKHZAN [8]</td><td>Yes</td><td>Yes</td><td>Yes</td></tr><tr><td>Ours</td><td>Yes</td><td>Yes</td><td>Yes</td></tr></table>

## C. Layout Analysis

Line detection is a key step in historical document processing. Kiessling [10] proposed a modular system that predicts baselines and regions with a neural network and then computes line polygons. This system, available in the kraken engine, handles curved and rotated lines and has been evaluated on Arabic-script manuscripts. We use it as the starting point for our line detector.

## D. Word Spotting

Word spotting retrieves the occurrences of a query in document images without full transcription. The attributeembedding framework of Almazan´ et al. [13] introduced the Pyramidal Histogram of Characters (PHOC), a fixed-length binary code that records which characters appear in which relative part of a word. Because a PHOC can be computed both from a word image and from a query string, images and strings live in a common space and retrieval reduces to a nearest-neighbour search. PHOCNet [14] replaced the handcrafted features with a convolutional network and became the standard query-by-string baseline. Ctrl-F-Net [15] extended this to segmentation-free spotting on full pages by jointly proposing word regions and embedding them. HWNet v3 [17] learns a joint embedding of word images and text supporting both retrieval and lexicon-based recognition, and Papazis et al. [16] re-rank retrieved word images using semantic embeddings from language models. These methods give strong results, but they all rely on word-level boxes for training or evaluation. Our approach needs only line-level annotation and recovers the word position from the frame alignment of the recognizer. In Section V-E we compare against a PHOC baseline that is given oracle word boundaries at test time, as an upper bound for the stronger-supervision family.

## III. DATASET

## A. Source Material

The pages were selected carefully from historical Persian books so that the dataset covers different genres, scripts, and layouts. The sources include poetry and prose. Poetry pages often use a two-column layout with separated hemistichs, while prose pages have long, continuous lines. The pages also vary in writing style, ink color (for example, red headings), decoration, and state of preservation. Some pages contain illuminated frames, cloud-shaped gilded backgrounds around the text, stains, and faded strokes. All pages were scanned at high quality and stored in JPG format.

TABLE II  
OVERALL STATISTICS OF THE DATASET.
<table><tr><td>Pages</td><td>Lines</td><td>Words</td><td>Characters</td></tr><tr><td>223</td><td>3,678</td><td>37,631</td><td>130,630</td></tr></table>

TABLE III

PAGE-LEVEL SPLIT OF THE DATASET USED FOR TRAINING AND EVALUATION.
<table><tr><td>Split</td><td>Pages</td><td>Lines</td><td>Words</td></tr><tr><td>Train</td><td>144</td><td>2,329</td><td>24,301</td></tr><tr><td>Validation</td><td>35</td><td>614</td><td>6,079</td></tr><tr><td>Test</td><td>44</td><td>735</td><td>7,251</td></tr><tr><td>Total</td><td>223</td><td>3,678</td><td>37,631</td></tr></table>

## B. Annotation Protocol

Annotation was carried out by 11 annotators using the Transkribus platform<sup>1</sup>. For each page the annotators marked three levels of information: text regions, line regions within each text region, and the transcription of each line. Fig. 1 shows an example. Each transcription was entered carefully to match the written text, and the annotations were exported as XML files. We did not draw word-level boxes. This choice kept the annotation cost manageable, and, as shown in Section IV, word positions can still be recovered by the model.

In addition to the per-line annotation, each page carries five page-level attributes that were assigned by visual inspection: genre (poetry, prose, mixed), script style (shekasteh, nastaliq, naskh), degradation (none, light, moderate), illumination or tazhib (none, border, both border and gilded cloud background), and a subjective reading difficulty (easy, moderate, hard) reflecting how hard a trained reader finds the hand to decipher. These attributes are released with the dataset and are used in Section V-G to report performance by condition.

## C. Statistics and Splits

The dataset contains 223 pages, 3,678 lines, 37,631 words, and 130,630 characters (Table II). After the normalization described below, 3,656 lines carry a non-empty transcription; the remaining lines are annotated regions whose content reduces to punctuation or decoration only, and they are excluded from training and evaluation. On average a page has about 16.4 lines and a line has about 10.3 words. We split the data at the page level into training, validation, and test sets, so no page appears in more than one set. The split is given in Table III.

## D. Text Normalization

Some symbols appear rarely in the transcriptions and make training harder without helping search. Before training, we normalize the text by removing diacritics (short vowel marks and tashdid), mapping the different forms of alef to a single form, mapping the Arabic yeh and kaf to their Persian counterparts, replacing the zero-width non-joiner with a space, and removing punctuation and other rare symbols. The same normalization is applied to the search queries, so a user can type a word in its plain form and still match the manuscript.

TABLE IV  
DISTRIBUTION OF STRUCTURAL QUANTITIES IN THE DATASET.
<table><tr><td>Quantity</td><td>Mean</td><td>Std</td><td>Median</td><td>Min</td><td>Max</td></tr><tr><td>Lines per page</td><td>16.4</td><td>6.3</td><td>15</td><td>1</td><td>30</td></tr><tr><td>Words per line</td><td>10.3</td><td>4.2</td><td>9</td><td>1</td><td>22</td></tr><tr><td>Characters per line</td><td>35.7</td><td>14.6</td><td>28</td><td>3</td><td>78</td></tr><tr><td>Characters per word</td><td>3.5</td><td>1.5</td><td>3</td><td>1</td><td>15</td></tr></table>

TABLE V

FREQUENCY OF THE LEAST FREQUENT PERSIAN LETTERS AFTER NORMALIZATION. COUNTS ARE OVER 130,628 CHARACTERS.
<table><tr><td>Letter</td><td>Unicode</td><td>Count</td><td>Share (%)</td></tr><tr><td>zhe</td><td>U+0698</td><td>23</td><td>0.018</td></tr><tr><td>zal</td><td>U+0630</td><td>369</td><td>0.282</td></tr><tr><td>se</td><td>U+062B</td><td>394</td><td>0.302</td></tr><tr><td>za</td><td>U+0638</td><td>396</td><td>0.303</td></tr><tr><td>ghayn</td><td>U+063A</td><td>474</td><td>0.363</td></tr><tr><td>pe</td><td>U+067E</td><td>597</td><td>0.457</td></tr><tr><td>zad</td><td>U+0636</td><td>835</td><td>0.639</td></tr><tr><td>che</td><td>U+0686</td><td>975</td><td>0.746</td></tr><tr><td>sad</td><td>U+0635</td><td>1,202</td><td>0.920</td></tr><tr><td>gaf</td><td>U+06AF</td><td>1,573</td><td>1.204</td></tr></table>

## E. Distributional Analysis

A dataset description is only useful for comparison if the distribution of its content is known. Table IV summarizes the structural quantities and Fig. 2 shows their histograms.

The distribution of words per line is clearly bimodal, with one mode near 7-8 words and a second near 15-16. This directly reflects the genre mixture: poetry pages use a two-column layout in which each annotated line is a single hemistich, while prose pages run the full width of the text block. Any method evaluated on this dataset therefore has to cope with a factor-of-two variation in line length, and results averaged over all lines mix two rather different regimes. Words are short, with a median of 3 characters and a mean of 3.5, which is important for retrieval: as shown in Section V-F, short queries are the dominant source of false positives.

The character frequency distribution (Fig. 3, Table V) spans nearly three orders of magnitude. The most frequent letter, alef, occurs about 17,000 times, while zhe occurs only 23 times in the whole dataset. Seven letters account for less than 0.5% of all characters each. This long tail is a practical constraint on any recognizer trained on this data: a character seen 23 times cannot be learned reliably, and queries containing it should be expected to fail. We report this explicitly so that future work can separate genuine modelling improvements from differences in how rare characters are handled.

Table VI and Fig. 4 give the composition by page attribute. Poetry accounts for 133 of the 223 pages and prose for 84.

![](images/b2df50f60c58feafa7ddb2524c25b39577f5e7687b86ead217f213272257c3ea.jpg)

![](images/6ee09d5299fe0ecd245a59b6bfdf213c40bb2e15ca61c5902759fe3869bc6abf.jpg)  
Fig. 1. An example page from the dataset (Majma al-Bayan). Left: the original scanned page. Right: the annotation made in Transkribus, with text regions in blue, line regions in green, and the line transcriptions in red. The page contains a decorated frame, red rubrics, and a two-column poetry block, which illustrate the layout variety of the dataset.

![](images/ed27acf71307c5472777bd00c2bb71ba29944cb06166857e41519b6a3c0b56f5.jpg)

![](images/3de1b84448cfc694266a857bbf99980350b420e773877e9484fda8a429918448.jpg)

![](images/774de39094dc922361613ace13ae1c1581c1901967665c09388e959ac408214e.jpg)  
Fig. 2. Distribution of words per line (left), characters per word (middle), and lines per page (right). The bimodal shape of the first and third histograms comes from the mixture of two-column poetry pages and full-width prose pages.

The dominant script is shekasteh (173 pages), followed by nastaliq (48) and naskh (2). Shekasteh is the hardest of the three to read, since letters are heavily joined and many dots are omitted, and its dominance is a deliberate choice: it is the script in which a large part of the surviving Persian administrative and literary manuscript record is written, and it is under-represented in existing datasets. One hundred and thirty-nine pages carry some illumination, and 99 show some degradation. Two categories are very small, naskh with 2 pages and moderate degradation with 3, and results conditioned on them should not be treated as reliable.

## IV. METHODOLOGY

The proposed pipeline has three stages: line detection, recognition, and posterior-based search (Fig. 5). All models were trained with the kraken engine<sup>2</sup>.

## A. Line Detection

For line detection we fine-tune the baseline and layout analysis model (BLLA) of Kiessling [10]. The network predicts baseline and region heatmaps for the page, and the baselines are then turned into line polygons. We start from the default pretrained BLLA weights and fine-tune them on the line annotations of our training set. Each detected line is cropped and rectified into a straight line image for the recognizer.

![](images/c0610cd6f8e8a55ef42fe63a564fd63740ed8e8f24811ec3a29639b6b97d7422.jpg)  
Fig. 3. Character frequency in the dataset after normalization, on a logarithmic scale. The distribution spans almost three orders of magnitude, from alef with about 17,000 occurrences to zhe with 23.

![](images/e073860e29daf13cf57210195bc6de81ab7f3133730742078ff46d433aaf157b.jpg)

![](images/7425ca40852424edd5472b0b1fb12dbff2df078b51c64447485bb9fba95dfdac.jpg)

![](images/26203a0d8ce44eea1fefa53b7dae9cb3263db4ababa81a75fb87542e8db4d25d.jpg)

![](images/d6d535ab75415a3a766b68ee6bd670aed78ee4aeb6946c98b290a077a70526cd.jpg)

![](images/61df75f577dca5f49ae661821bad747a3774392768f878c6d7339b2476dc9340.jpg)

Fig. 4. Composition of the dataset by genre, script style, degradation, illumination (tazhib), and subjective reading difficulty, in pages. The difficulty label coincides with the shekasteh script class, since that script drives the reading difficulty on almost every page where it appears.  
![](images/f1e515bc0d10b6953bd113ca1c94ec7f8628cd7dcde02843cacc48460581d38c.jpg)  
Fig. 5. Overview of the proposed pipeline. Text lines are detected on the page, each line image is split into frames by the recognizer, and the recognizer outputs a frame-by-character posterior matrix (including the CTC blank). Word spotting is performed on this matrix instead of on the decoded text, and the matched frames give the position of the word in the line and the page.

## B. Recognition

For recognition we fine-tune the pretrained all\_arabic model<sup>3</sup>, a CRNN trained with CTC loss on Arabic-script text. The model follows the usual convolutional-recurrent design [12]. A line image is rescaled to a fixed height while its width is kept free. Convolutional layers downsample the image along its width and produce a sequence of $T$ frames, where each frame describes a thin vertical slice of the line. A bidirectional LSTM reads this sequence in both directions, and a linear layer with softmax gives, for every frame, a probability distribution over the character set plus a special blank symbol. CTC training needs only the line transcription as a label, so no character or word boxes are required. Because Persian is written from right to left while frames are ordered from left to right in the image, kraken applies the Unicode bidirectional algorithm to put the labels in display order. We apply the same reordering to the queries.

TABLE VI  
COMPOSITION OF THE DATASET BY GENRE, SCRIPT STYLE, DEGRADATION, ILLUMINATION, AND SUBJECTIVE READING DIFFICULTY.
<table><tr><td>Attribute</td><td>Value</td><td>Pages</td><td>Lines</td><td>Words</td></tr><tr><td>Genre</td><td>poetry</td><td>133</td><td>2,535</td><td>22,032</td></tr><tr><td></td><td>prose</td><td>84</td><td>1,033</td><td>14,660</td></tr><tr><td></td><td>mixed</td><td>6</td><td>88</td><td>939</td></tr><tr><td>Style</td><td>shekasteh</td><td>173</td><td>3,000</td><td>32,878</td></tr><tr><td></td><td>nastaliq</td><td>48</td><td>633</td><td>4,443</td></tr><tr><td></td><td>naskh</td><td>2</td><td>23</td><td>310</td></tr><tr><td>Degradation</td><td>none</td><td>124</td><td>2,293</td><td>19,639</td></tr><tr><td></td><td>light</td><td>96</td><td>1,328</td><td>17,545</td></tr><tr><td></td><td>moderate</td><td>3</td><td>35</td><td>447</td></tr><tr><td>Tazhib</td><td>none</td><td>84</td><td>973</td><td>13,944</td></tr><tr><td></td><td>border</td><td>117</td><td>2,310</td><td>21,464</td></tr><tr><td></td><td>both</td><td>22</td><td>373</td><td>2,223</td></tr><tr><td>Difficulty</td><td>easy</td><td>30</td><td>301</td><td>2,690</td></tr><tr><td></td><td>moderate</td><td>20</td><td>355</td><td>2,063</td></tr><tr><td></td><td>hard</td><td>173</td><td>3,000</td><td>32,878</td></tr></table>

Both models were trained with a learning rate of $1 0 ^ { - 4 }$ , a batch size of 8, and 40 epochs.

## C. The Posterior Matrix

For a line image with T frames and a character set with K characters, the recognizer outputs a matrix

$$
Y \in [ 0 , 1 ] ^ { T \times ( K + 1 ) } , \qquad \sum _ { c } y _ { t } ( c ) = 1 ,\tag{1}
$$

where $y _ { t } ( c )$ is the probability of symbol c at frame t and the extra column is the blank. A normal OCR system reduces $Y$ to text by picking arg $\operatorname* { m a x } _ { c } y _ { t } ( c )$ at every frame and then removing repeated symbols and blanks. We keep Y in full. In practice we read the values of the layer before the final decoding step, i.e., the softmax output of the network, and store one matrix per detected line.

## D. Scoring a Query Against the Posterior

Let the normalized query be $q = ( q _ { 1 } , \ldots , q _ { L } )$ . As in CTC, we insert blanks around and between its characters to get the extended sequence

$$
\boldsymbol { q } ^ { \prime } = ( \emptyset , q _ { 1 } , \emptyset , q _ { 2 } , \dots , \emptyset , q _ { L } , \emptyset ) ,\tag{2}
$$

of length $2 { \cal L } + 1$ , where ∅ is the blank. The query can appear anywhere in the line, so the alignment is allowed to start and end at any frame.

To make scores comparable across lines, we measure each frame relative to the best symbol at that frame:

$$
r _ { t } ( c ) = \log y _ { t } ( c ) - { \underset { k } { \operatorname* { m a x } } } \log y _ { t } ( k ) \ \leq \ 0 .\tag{3}
$$

If c is the top symbol at frame $t ,$ then $r _ { t } ( c ) = 0 ;$ otherwise it is negative, and the more the model prefers another symbol, the more negative it becomes.

We then find the best CTC alignment of $q ^ { \prime }$ with a Viterbi recursion. Let $\delta _ { t } ( s )$ be the best score of a path that ends at frame t in state s of $q ^ { \prime } { : }$

$$
\delta _ { t } ( s ) = r _ { t } ( q _ { s } ^ { \prime } ) + \operatorname* { m a x } \bigl \{ \delta _ { t - 1 } ( s ) , \delta _ { t - 1 } ( s - 1 ) , \delta _ { t - 1 } ( s - 2 ) \bigr \} ,\tag{4}
$$

where the jump from $s - 2$ is allowed only when $q _ { s } ^ { \prime }$ is not blank and $q _ { s } ^ { \prime } \ne q _ { s - 2 } ^ { \prime } ,$ , as in standard CTC. To let the word start at any frame, the first two states $( s = 1$ and $s = 2 )$ may also start fresh with a previous score of zero. The final score of the query in the line is

$$
S ( q ) = \operatorname* { m a x } _ { t } \operatorname* { m a x } \big \{ \delta _ { t } ( 2 L ) , \delta _ { t } ( 2 L + 1 ) \big \} ,\tag{5}
$$

and we normalize it to a per-character log score,

$$
s ( q ) = \frac { S ( q ) } { L } \ \leq \ 0 ,\tag{6}
$$

equivalently a per-character confidence $C ( q ) = \exp ( s ( q ) ) \in$ $( 0 , 1 ]$ . Dividing by L keeps long queries from being penalized only for their length. A line is reported as containing the query when $s ( q ) \geq \tau .$ . The selection of τ on the validation set is described in Section V-C. Several non-overlapping matches can be returned from the same line.

## E. Why Posterior Scoring Helps: An Intuitive View

It is easiest to see the benefit with an example. Suppose the manuscript contains the word ketab (“book”), which ends with the letter be. Because the dot is faint, the recognizer is not sure about the last letter. At that frame it gives be a probability of 0.45 and $p e$ a probability of 0.40.

A normal OCR system picks the highest value at each frame and writes down the result. If the dot is read slightly differently and $p e$ wins, the output becomes a different string, and a text search for ketab finds nothing. The word is lost even though the model was almost right.

Posterior scoring does not force this early choice. It asks a different question: “how likely is it that the word ketab is written here?” Since be still has a high probability at that frame, the query still receives a high score and the line is returned. In other words, the uncertainty of the model is kept until the end, and small mistakes reduce the score a little instead of removing the word completely. This matters most for Persian, where many letters differ only by dots.

The same idea explains why recall increases. A historian mainly needs the correct occurrences to appear among the results; the recognizer does not have to be perfect at every character. The price is that some visually similar words also receive a reasonable score, which can lower precision. We quantify this trade-off in Section V-F.

## F. Recovering the Word Position

The alignment also tells us where the word is. The best path gives the first frame $t _ { s }$ and the last frame $t _ { e }$ that are assigned to the query characters. Since each frame covers a fixed slice of the line image, the horizontal extent of the word in a line image of width W is approximately

$$
x _ { \mathrm { s t a r t } } = \frac { t _ { s } } { T } W , \qquad x _ { \mathrm { e n d } } = \frac { t _ { e } + 1 } { T } W .\tag{7}
$$

The vertical extent is taken from the detected line. By mapping these coordinates back through the line polygon, we obtain the word location on the page. In this way a model trained only with line annotations produces word-level results. Fig. 7(c)-(d) shows two examples of the recovered box.

## G. Exact-Match Baseline

To measure the effect of the posterior search, we compare it with a standard decode-then-search baseline. Here the line is decoded with the argmax path, and a hit is reported when the query appears as a word in the decoded text. The position is taken from the frames of the decoded characters, in the same way as above.

## H. PHOC Query-by-String Baseline

As a representative of the stronger-supervision family we implement a PHOC attribute-embedding baseline in the style of Almazan´ et al. [13] and PHOCNet [14]. A word is encoded as a binary pyramidal histogram over five levels (1, 2, 3, 4, 5 splits) of a 42-symbol alphabet, giving a 630-dimensional target. A convolutional network with four blocks of two $3 \times 3$ convolutions followed by max pooling, and two fully connected layers, maps a word image to this space and is trained with a binary cross-entropy loss. At query time the PHOC of the query string is computed analytically and compared to every word embedding by cosine similarity; a line is scored by the maximum similarity over its words.

This baseline needs word images, which our dataset does not provide. We obtain them by CTC forced alignment of the ground-truth transcription against the posterior matrix, which yields a frame span per character and therefore a box per word. This is done on both the training and the test pages. Consequently the PHOC baseline receives information that the proposed method never uses, namely the correct transcription of every test line, and its result should be read as an upper bound for word-box-supervised methods on this data rather than as a like-for-like comparison.

## V. EXPERIMENTS AND RESULTS

## A. Evaluation Protocol

Line detection. A predicted line is matched to a ground-truth line when their intersection over union (IoU) is at least 0.3, with each ground-truth line matched at most once. We report precision, recall, F1, and the mean IoU of the matched lines on the 44 test pages (735 lines).

Word spotting. For each test page, ten words of at least three characters were chosen at random from the ground truth, giving 396 distinct query words. Every query is searched in all lines of all test pages, producing roughly $2 . 8 \times 1 0 ^ { 5 }$ query-line pairs of which 1,430 are relevant. A retrieved result is counted as correct when it falls in a line that contains the query word in its ground-truth transcription. We report precision, recall, and F1 over all queries at the operating threshold τ.

TABLE VII  
LINE DETECTION RESULTS ON THE TEST SET (44 PAGES, 735 LINES) AT AN IOU THRESHOLD OF 0.3.
<table><tr><td>Model</td><td>Precision</td><td>Recall</td><td>F1</td><td>Mean IoU</td></tr><tr><td>Base model</td><td>0.751</td><td>0.841</td><td>0.793</td><td>0.412</td></tr><tr><td>Our model</td><td>0.867</td><td>0.920</td><td>0.892</td><td>0.534</td></tr></table>

We do not use character or word error rate here. Our goal is to find words, not to produce a perfect transcription, and a model with a moderate error rate can still be useful for retrieval if the correct words receive high scores.

Models. “Base model” refers to the pretrained BLLA and all\_arabic models without fine-tuning. “Our model” refers to the same models after fine-tuning on our training set.

## B. Line Detection Results

Table VII shows the line detection results. Fine-tuning improves all measures. Precision increases from 0.751 to 0.867 and recall from 0.841 to 0.920, and the F1 score rises from 0.793 to 0.892. The mean IoU also increases from 0.412 to 0.534. The mean IoU values are moderate because the line polygons produced from baselines do not have exactly the same extent as the manually drawn line regions, especially for lines with tall ascenders, long descenders, or decorated backgrounds. A threshold of 0.3 is still enough to decide whether the correct line was found, which is what matters for retrieval.

## C. Threshold Selection

The decision threshold τ is selected by maximizing F1 on the 35 validation pages and is then frozen before the test set is touched. This gives $\tau ^ { * } = - 0 . 8 0$ in per-character log score, equivalently a per-character confidence of 0.45. At this threshold the method reaches $\mathrm { F 1 = 0 . 5 7 2 }$ on validation and $\mathrm { F 1 = 0 . 5 5 8 }$ on test. The best threshold that could have been chosen with knowledge of the test set is $\tau = - 0 . 8 8 .$ , giving F1 $= 0 . 5 6 5$ . The gap of 0.007 F1 between the validation-selected and the oracle threshold indicates that $\tau$ transfers well across pages, which matters in practice because a user of the released system cannot tune it.

Fig. 6 shows precision, recall, and F1 as functions of $\tau$ on the test set, together with the precision-recall curve. The F1 curve is flat within about $\pm 0 . 3 \ \mathrm { o f } \ \tau ^ { * }$ , so the exact value is not critical. The two ends of the curve are more informative than the maximum. Precision rises steeply only in the last fraction of the range: reaching precision 0.90 requires $\tau = - 0 . 0 2$ which retrieves almost nothing (recall 0.009). Conversely, recall 0.90 is available at $\tau = - 2 . 4 2$ but at precision 0.071, meaning roughly thirteen results must be inspected per correct hit. Table VIII lists the three operating points.

TABLE VIII  
OPERATING POINTS OF POSTERIOR SEARCH ON THE TEST SET. $\tau ^ { * }$ ISSELECTED BY MAXIMIZING F1 ON THE VALIDATION SET AND IS NOTTUNED ON TEST.
<table><tr><td>Operating point</td><td>T</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>Precision  $\geq 0 . 9 0$ </td><td>-0.02</td><td>0.929</td><td>0.009</td><td>0.018</td></tr><tr><td> $\tau ^ { * }$  (max F1 on val)</td><td>-0.80</td><td>0.570</td><td>0.548</td><td>0.558</td></tr><tr><td>Recall  $\geq 0 . 9 0$ </td><td>-2.42</td><td>0.071</td><td>0.900</td><td>0.132</td></tr></table>

TABLE IX

WORD SPOTTING RESULTS ON THE TEST SET (44 PAGES, 735 LINES, 7,251 WORDS) WITH 396 QUERY WORDS. POSTERIOR SEARCH USES  
$\tau ^ { * } = - 0 . 8 0$ SELECTED ON THE VALIDATION SET.
<table><tr><td>Model</td><td>Search</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>Base model</td><td>Exact OCR</td><td>0.627</td><td>0.043</td><td>0.080</td></tr><tr><td></td><td>Posterior</td><td>0.288</td><td>0.218</td><td>0.248</td></tr><tr><td>Our model</td><td>Exact OCR</td><td>0.857</td><td>0.340</td><td>0.487</td></tr><tr><td></td><td>Posterior</td><td>0.570</td><td>0.548</td><td>0.558</td></tr></table>

This shape has a direct consequence for how the system should be deployed. The near-vertical precision curve at the right end means there is no threshold at which the method is both precise and useful; a historian must either accept a moderate precision around 0.57 at the F1 optimum, or work in a high-recall regime and filter visually. The latter is often acceptable, because judging a returned line image takes a second, whereas a missed occurrence is invisible.

## D. Word Spotting Results

Table IX reports the word spotting results for the two models and the two search strategies.

With the base model and exact matching, recall is only 0.043. The pretrained Arabic recognizer rarely produces Persian words exactly, so the search almost never succeeds, even though the few hits it finds are often correct (precision 0.627). Scoring against the posterior raises recall to 0.218 and the F1 score from 0.080 to 0.248, about three times higher. This shows that useful information is present in the posterior even when the decoded text is wrong.

Fine-tuning improves both strategies. With exact matching, our model reaches a precision of 0.857 but a recall of only 0.340: when the decoded word is correct it is almost always a real occurrence, but about two thirds of the occurrences are missed because at least one character was decoded incorrectly. With posterior search at the validation-selected threshold, recall increases to 0.548 and F1 from 0.487 to 0.558, while precision drops to 0.570. This is the expected trade-off, and Section V-F identifies where the lost precision goes.

## E. Comparison with a Query-by-String Method

Table X compares the proposed method with the PHOC baseline of Section IV-H. The PHOC network reaches F1 = 0.449 and $\mathrm { m A P = 0 . 2 6 6 }$ at line level, and $\mathrm { F 1 = 0 . 4 6 2 }$ and $\mathrm { m A P = 0 . 3 9 3 }$ at page level. Its precision at the F1 optimum is higher than ours (0.664 against 0.570), which is expected: it scores whole words against whole words and therefore does not fire on substrings. Its recall is much lower (0.339 against 0.548).

TABLE X  
COMPARISON WITH A PHOC ATTRIBUTE-EMBEDDING QUERY-BY-STRINGBASELINE ON THE TEST SET. THE PHOC BASELINE ADDITIONALLYRECEIVES ORACLE WORD BOUNDARIES OBTAINED BY FORCEDALIGNMENT OF THE GROUND-TRUTH TRANSCRIPTION, AND ISTHEREFORE AN UPPER BOUND FOR WORD-BOX-SUPERVISED METHODSRATHER THAN A LIKE-FOR-LIKE COMPARISON.
<table><tr><td>Level</td><td>Method</td><td>P</td><td>R</td><td>F1</td><td>mAP</td></tr><tr><td>Line</td><td>PHOC (oracle boxes)</td><td>0.664</td><td>0.339</td><td>0.449</td><td>0.266</td></tr><tr><td></td><td>Posterior (ours)</td><td>0.570</td><td>0.548</td><td>0.558</td><td>0.564</td></tr><tr><td>Page</td><td>PHOC (oracle boxes)</td><td>0.601</td><td>0.375</td><td>0.462</td><td>0.393</td></tr><tr><td></td><td>Posterior (ours)</td><td>0.601</td><td>0.639</td><td>0.619</td><td>0.639</td></tr></table>

The comparison should be read with two qualifications, both of which favour the baseline. First, it is given oracle word boundaries derived from the ground-truth transcription of the test lines, so its segmentation is perfect in a way no deployed system could be. Second, our implementation is a faithful but not exhaustively tuned PHOCNet; the original works train for far longer with heavier augmentation, and a fully tuned version would score higher. Even with these advantages the attribute embedding does not overtake posterior search at this data scale. We read this as a statement about data rather than about architecture: learning a 630-dimensional image-toattribute mapping from roughly 19,000 word crops of a single script family is a harder estimation problem than adapting an already-pretrained CTC recognizer, whose supervision is dense at every frame.

## F. Error Analysis

Aggregate scores hide which queries fail and why. Table XI breaks performance down by query property at $\tau ^ { * }$

Query length is by far the strongest factor. Queries of 3-4 characters reach precision 0.515, while queries of 5-6 characters reach 0.861 and queries of 7 or more reach 1.000. Recall moves in the opposite direction but far more weakly, from 0.574 to 0.444. Since the median word in the dataset is only 3 characters long (Table IV), short queries dominate the query set and drag the aggregate precision down. The mechanism is straightforward: a three-character pattern has many near-matches inside a 36-character line, and the freestart, free-end alignment is allowed to find any of them. A practical implication is that the reported aggregate understates the usefulness of the system for the queries a historian actually issues, which are typically names and technical terms of five characters or more.

The proportion of dotted letters in the query has almost no effect: F1 moves from 0.578 to 0.538 as the dotted share goes from below 25% to above 50%. This is worth stating because it runs against the intuition motivating the method. Posterior scoring was introduced precisely to survive dot confusions, and the flat curve is evidence that it does: the confusions are absorbed by the score rather than converted into misses. The taxonomy below supports the same reading.

Table XII classifies the 635 false positives at $\tau ^ { * }$ . A little over half (53.4%) are cases where the query is a genuine substring of a longer word present in the line. This quantifies the limitation noted qualitatively in earlier drafts of this work: because the alignment may start and end at any frame, nothing prevents it from matching a prefix or infix. In Persian this is aggravated by clitics and by the prefixes be-, mi-, and na-, which turn many short content words into substrings of longer inflected forms. A further 45.0% are unrelated words, and only 1.6% are words sharing the query’s rasm (skeleton) but differing in dots. The dot-confusion failure that motivates the method is therefore almost absent from the error budget, while word-boundary ambiguity accounts for the majority of it. This points to a concrete and cheap improvement: requiring a blank or space-like frame at the two ends of the match should remove a large part of the false positives without affecting recall for whole-word queries.

![](images/db2d56dfd9ebfdcdbefec06d625f44379ad65e11d01afacb09c3a902436b630b.jpg)

![](images/a5443338f8668cac22f4a4d6a4c36cc4392cb41c9c2e191d30ccd03267d37b57.jpg)  
Fig. 6. Effect of the decision threshold τ on the test set. Left: precision, recall, and F1 as functions of $\tau ;$ the dashed line marks $\tau ^ { * } = - 0 . 8 0$ , selected by maximizing F1 on the validation set. Right: the corresponding precision-recall curve, with $\tau ^ { * }$ marked. Precision only rises sharply in the last fraction of the range, where recall has already collapsed.

TABLE XI  
WORD SPOTTING PERFORMANCE BY QUERY PROPERTY $\mu \mathrm { r } \tau ^ { * } = - 0 . 8 0$
<table><tr><td>Query group</td><td>Queries</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>Length 3-4</td><td>223</td><td>0.515</td><td>0.574</td><td>0.543</td></tr><tr><td>Length 5-6</td><td>144</td><td>0.861</td><td>0.524</td><td>0.651</td></tr><tr><td>Length 7+</td><td>17</td><td>1.000</td><td>0.444</td><td>0.615</td></tr><tr><td>Dotted ≤ 25%</td><td>118</td><td>0.580</td><td>0.577</td><td>0.578</td></tr><tr><td>Dotted 25-50%</td><td>175</td><td>0.555</td><td>0.567</td><td>0.561</td></tr><tr><td>Dotted &gt; 50%</td><td>91</td><td>0.540</td><td>0.535</td><td>0.538</td></tr></table>

TABLE XII

TAXONOMY OF THE 635 FALSE POSITIVES OF POSTERIOR SEARCH AT $\tau ^ { * } = - 0 . 8 0 .$
<table><tr><td>Cause</td><td>Count</td><td>Share (%)</td></tr><tr><td>Substring of a longer word</td><td>339</td><td>53.4</td></tr><tr><td>Unrelated word</td><td>286</td><td>45.0</td></tr><tr><td>Same rasm, different dots</td><td>10</td><td>1.6</td></tr></table>

Fig. 7 shows four representative cases. The two false negatives, (a) and (b), are both from heavily degraded or tightly written shekasteh lines. In (a) the query hadith occurs at the left edge of a line with large ink blots, and the posterior is dominated by the blank symbol across those frames, so no alignment accumulates a competitive score (−4.155). In (b) the query qadah is a short three-letter word inside a densely joined hemistich; the letters are present but the model distributes probability across several alternatives at every frame, leaving the per-character score at −3.472. Both illustrate the same failure mode: the method degrades when the recognizer is uncertain about the entire region, not when it is uncertain about one character.

The two true positives, (c) and (d), score near zero (−0.011 and −0.014), meaning the query characters were essentially the top choice at every aligned frame. The red boxes are the word locations recovered from the frame alignment, with no word-level supervision anywhere in the pipeline. In both cases the box encloses the correct word with a small horizontal margin, which is the accuracy a search interface needs in order to highlight a hit on the page.

## G. Performance by Page Condition

Table XIII reports word spotting performance on the test lines grouped by the page-level attributes of Section III.

Illumination has a clear and monotone effect. Lines on pages without tazhib reach $\mathrm { F 1 ~ = ~ 0 . 6 5 2 }$ , lines on pages with an illuminated border drop to 0.481, and lines on pages with both a border and a gilded cloud background drop further to 0.383, a loss of 0.269 F1 from the first group to the last. The cloud backgrounds are the more damaging of the two, since the gilding sits directly behind the text and reduces stroke contrast, whereas a border only affects the page margin. This is the single strongest conditioning factor we measured and it suggests that background suppression, rather than better sequence modelling, is where the next gain on this material lies.

The other two attributes behave counter-intuitively and we report them as such. Lines labelled hard score higher (F1 = 0.576) than lines labelled easy (0.507), and lines with light degradation score higher (0.621) than lines with none (0.475). Neither ordering should be read as evidence that degradation helps. Both are consequences of how the labels are distributed: the hard class contains 584 of the 733 test lines while easy contains only 59, so the two are not comparable samples, and the label was assigned by reading difficulty for a human rather than by any property the model is sensitive to. The same confound affects degradation, whose categories correlate with the source books and therefore with script style and illumination rather than varying independently. We keep these rows in the table for completeness and to make the confound visible, but the only conditioning conclusion we draw from this experiment is the one about illumination. Building a difficulty axis that is both balanced and independent of the other attributes would require either a larger test set or an automatically computed score, for example from blur, contrast, and baseline curvature; we leave this to future work on the dataset.

![](images/24845c2cc64377ef9de0af4af4487fe1f6e49e1f1b40ca5a6e06d1adb1f0252f.jpg)  
Fig. 7. Qualitative examples of posterior search. (a), (b): false negatives. In (a) the query hadith is lost in a line with heavy ink blots; in (b) the short query qadah sits in a densely joined shekasteh hemistich where the recognizer spreads probability over several candidates. (c), (d): true positives with the word location recovered from the CTC frame alignment shown in red. No word-level annotation was used at any stage; the boxes come from the alignment alone

TABLE XIII  
WORD SPOTTING PERFORMANCE ON THE TEST SET (733 LINES) GROUPED BY PAGE-LEVEL ATTRIBUTE, AT τ<sup>∗</sup> = −0.80. THE difficulty AND degradation GROUPS ARE STRONGLY IMBALANCED AND CONFOUNDED WITH SCRIPT STYLE; SEE THE TEXT.
<table><tr><td>Attribute</td><td>Value</td><td>Lines</td><td>P</td><td>R</td><td>F1</td></tr><tr><td rowspan="3">Tazhib</td><td>none</td><td>197</td><td>0.626</td><td>0.680</td><td>0.652</td></tr><tr><td>border</td><td>446</td><td>0.522</td><td>0.446</td><td>0.481</td></tr><tr><td>both</td><td>90</td><td>0.412</td><td>0.357</td><td>0.383</td></tr><tr><td rowspan="3">Difficulty</td><td>easy</td><td>59</td><td>0.518</td><td>0.496</td><td>0.507</td></tr><tr><td>moderate</td><td>90</td><td>0.412</td><td>0.357</td><td>0.383</td></tr><tr><td>hard</td><td>584</td><td>0.585</td><td>0.567</td><td>0.576</td></tr><tr><td rowspan="3">Degradation</td><td>none</td><td>456</td><td>0.512</td><td>0.443</td><td>0.475</td></tr><tr><td>light</td><td>257</td><td>0.607</td><td>0.635</td><td>0.621</td></tr><tr><td>moderate</td><td>20</td><td>0.632</td><td>0.590</td><td>0.610</td></tr></table>

## H. Discussion

Four points stand out from the results.

First, fine-tuning on even a modest amount of historical Persian data is necessary. A model trained on general Arabicscript text is not enough for historical Persian manuscripts, both for finding lines and for reading them.

Second, keeping the full posterior matrix is a simple change that gives a clear gain in recall and F1 without any extra annotation or retraining, and it outperforms a PHOC attribute embedding even when that baseline is handed oracle word boundaries.

Third, the error budget is not where the motivation predicted. Dot confusions, which the method was designed to survive, account for 1.6% of false positives, while wordboundary ambiguity accounts for 53.4%. The method solves the problem it was built for, and the remaining precision loss is a different problem with a different and likely cheaper fix.

Fourth, illumination is the dominant page-level factor, costing 0.269 F1 between clean pages and pages with gilded cloud backgrounds.

The approach also has limitations. The word position is estimated from frames, so its horizontal boundaries are approximate, and we could not measure word-level localization accuracy directly because the dataset has no word boxes; the boxes in Fig. 7 are therefore illustrative rather than quantified. The precision of posterior search is limited by substring matches, which could be addressed by requiring blank frames at the match boundaries or by a lexicon or language model. Our test set of 44 pages is small enough that conditioned results on minority categories are unreliable. Finally, the subjective difficulty label did not prove to be a useful conditioning variable and should be replaced by an automatically computed one in future releases.

## VI. CONCLUSION

We introduced a new dataset of historical Persian handwriting with 223 pages, 3,678 lines, and 37,631 words, collected from diverse books of poetry and prose and annotated at the region, line, and text level, together with page-level attributes and a distributional analysis of its content. We also proposed a word spotting baseline that is trained only with line-level labels. The method detects lines with a fine-tuned BLLA model, reads them with a fine-tuned CRNN, and searches the CTC posterior matrix instead of the decoded text. This makes the search tolerant to confusions between similar letters and gives the word position within the line and the page. On the test set, the fine-tuned line detector reaches an F1 of 0.892, and posterior search improves word spotting F1 from 0.487 to 0.558 over exact matching at a threshold selected on held-out validation data, ahead of a PHOC attributeembedding baseline with oracle word boundaries. An error analysis shows that the dominant remaining failure is substring matching rather than character confusion, and that illuminated backgrounds are the strongest page-level factor. We hope the dataset and baseline will support further work on searching historical Persian manuscripts, for example with transformerbased recognizers [11], [18], learned word embeddings [17], and re-ranking methods [16].

## DATA LICENSE

The data will be distributed under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license for research purposes. The page images are derived from digitized historical manuscripts, and any use of them should also respect the terms of the holding institutions.

## REFERENCES

[1] J. Sadri, M. R. Yeganehzad, and J. Saghi, “A novel comprehensive database for offline Persian handwriting recognition,” Pattern Recognit., vol. 60, pp. 378-393, 2016.

[2] P. Jafarzadeh, P. Choobdar, and V. Mohammadi Safarzadeh, “Khayyam offline Persian handwriting dataset,” arXiv preprint arXiv:2406.01025, 2024.

[3] H. Khosravi and E. Kabir, “Introducing a very large dataset of handwritten Farsi digits and a study on their varieties,” Pattern Recognit. Lett., vol. 28, no. 10, pp. 1133-1141, 2007.

[4] M. Ziaratban, K. Faez, and F. Bagheri, “FHT: An unconstraint Farsi handwritten text database,” in Proc. 10th Int. Conf. Document Anal. Recognit. (ICDAR), 2009, pp. 281-285.

[5] U.-V. Marti and H. Bunke, “The IAM-database: An English sentence database for offline handwriting recognition,” Int. J. Document Anal. Recognit., vol. 5, pp. 39-46, 2002.

[6] M. Kassis, A. Abdalhaleem, A. Droby, R. Alaasam, and J. El-Sana, “VML-HD: The historical Arabic documents dataset for recognition systems,” in Proc. 1st Int. Workshop Arabic Script Anal. Recognit. (ASAR), 2017, pp. 11-14.

[7] R. Basharat and M. U. Ali, “Urdu Katib handwritten dataset: A historical document dataset for offline Urdu handwritten text recognition with CRNN-based baseline evaluation,” arXiv preprint arXiv:2606.19139, 2026.

[8] J. P. Allen et al., “OpenITI MAKHZAN: An open annotated dataset of Arabic, Persian, Ottoman Turkish, and Urdu print and manuscript data,” J. Open Humanities Data, vol. 12, art. 69, pp. 1-12, 2026, doi: 10.5334/johd.465.

[9] M. Jampour, S. Farridnejad, A. KarimiSardar, K. Champour, A. Aghaee Meybodi, F. Hematzadeh, and L. Paul, “MPHD: Multi-purpose Persian handwriting dataset for text recognition, line segmentation, writer identification, and handwritten analysis,” preprint, 2026, doi: 10.17632/2fcw6cd9gd.1.

[10] B. Kiessling, “A modular region and text line layout analysis system,” in Proc. 17th Int. Conf. Frontiers Handwriting Recognit. (ICFHR), 2020, pp. 313-318.

[11] A. Chan, A. Mijar, M. Saeed, C.-W. Wong, and A. Khater, “HATFormer: Historic handwritten Arabic text recognition with transformers,” arXiv preprint arXiv:2410.02179, 2024.

[12] J. Wang and X. Hu, “Gated recurrent convolution neural network for OCR,” in Adv. Neural Inf. Process. Syst. (NIPS), 2017, pp. 335-344.

[13] J. Almazan, A. Gordo, A. Forn´ es, and E. Valveny, “Word spotting and´ recognition with embedded attributes,” IEEE Trans. Pattern Anal. Mach Intell., vol. 36, no. 12, pp. 2552-2566, 2014.

[14] S. Sudholt and G. A. Fink, “PHOCNet: A deep convolutional neural network for word spotting in handwritten documents,” in Proc. 15th Int. Conf. Frontiers Handwriting Recognit. (ICFHR), 2016, pp. 277-282.

[15] T. Wilkinson, J. Lindstrom, and A. Brun, “Neural Ctrl-F: Segmentation-¨ free query-by-string word spotting in handwritten manuscript collections,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 4433- 4442.

[16] S. Papazis, A. P. Giotis, and C. Nikou, “Enhancing keyword spotting via NLP-based re-ranking: Leveraging semantic relevance feedback in the handwritten domain,” Electronics, vol. 14, no. 14, art. 2900, 2025.

[17] P. Krishnan, K. Dutta, and C. V. Jawahar, “HWNet v3: A joint embedding framework for recognition and retrieval of handwritten text,” Int J. Document Anal. Recognit., vol. 26, pp. 401-417, 2023.

[18] S. Sharma, M. Flammini, and F. Simonetta, “TrOCR for medieval HTR: A systematic ablation study with cross-dataset validation,” arXiv preprint arXiv:2606.24302, 2026.