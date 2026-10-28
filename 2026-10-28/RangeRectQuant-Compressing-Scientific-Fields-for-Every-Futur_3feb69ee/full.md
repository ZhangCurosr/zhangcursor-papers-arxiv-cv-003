# RangeRectQuant: Compressing Scientific Fields for Every Future Rectangular Sum

Jiahao Zhang, Xinling Wen, Yu Chen<sup>∗</sup> Zhengzhou University of Aeronautics <sup>∗</sup>Corresponding author: chenyu@zua.edu.cn

## Abstract

Scientific rasters are often stored before the regions to be measured are known. Independent pixel-error bounds can force exact integer reconstruction when one absolute error budget covers every future rectangle. Coordinated changes allow every pixel to change while every rectangular sum remains accurate. A prefix sum accumulates values from the top-left corner. Four such sums determine any rectangular total. RangeRectQuant preserves the pixel range and designated exact samples while controlling errors in prefix sums. At each row position, we characterize exactly which errors let the remaining pixels meet these constraints. These errors form a single integer interval, even when many completions exist. A backward pass finds its endpoints; a forward pass uses them to choose quantized pixels. The resulting field meets the prescribed error budget for every axis-aligned rectangle, with linear work and workspace proportional to one row. Experiments on geographically held-out builtsurface and water-occurrence maps show substantial storage savings over alternatives meeting the same requirements. Data producers encode once; later users measure arbitrary rectangles without the source, re-encoding, or query-specific correction.

## Introduction

Scientific rasters are archived before future measurement regions are known. Later users total values over neighborhoods, tiles, or other rectangles. Built-surface maps record building area per pixel. We seek one compressed field that keeps every rectangular sum within a prescribed error budget.

A natural approach gives each pixel the same independent error bound. Errors from many pixels can accumulate in the same direction. Certifying a small regional budget this way can force exact integer reconstruction (Proposition 1). Coordinated errors allow every pixel to change: turning ones into alternating zeros and twos changes any rectangular sum by at most one (Figure 1).

A prefix sum adds values from the top-left corner to a pixel; four such sums determine any rectangular total [1, 2]. Bounded prefix errors give an area-independent guarantee. Pixel ranges complicate coordination: zero cannot decrease, the maximum cannot increase, and a current choice can prevent the row from finishing legally.

RangeRectQuant looks ahead before committing a pixel (Figure 2). All prefix-error states that can reach a legal row completion form a single integer interval. A backward pass finds its endpoints; a forward pass uses it to choose compressible pixels. Linear work guarantees every rectangle, including over 275 billion in a 1024 × 1024 field.

![](images/c696b3dfcc5372a0510e941cd144a1807727695693203da02329d61dfea21da6.jpg)  
Figure 1: Coordination on a field of ones, with valid range zero to two. Every pixel changes; every rectangle error has magnitude at most one. Prefix errors stay between minus one and zero.

## Related Work

Pointwise distortion is a central target of scientific compression [3]. Prediction and interpolation methods improve storage rates while controlling individual reconstruction errors [4–7]. Quantization-error coding [8] and integer representations for astronomical images [9] also improve rate–precision tradeofs. A uniform pixel tolerance gives each sample the same freedom to change. Requiring every regional sum to share one small budget can remove that freedom entirely (Proposition 1).

Specified downstream quantities bring compression closer to the analysis users perform [10]. These include derived measurements bounded through field norms [11,12], calculations preserved through tailored pixel tolerances [13, 14], and selected ranges, regions, or topological features [15, 16]. Query-aware representations optimize meansquared range-query error [17] or predict bias and uncertainty [18]. Archived rasters pose a diferent requirement: the measurement regions are unknown at encoding time, and every later rectangle must satisfy a deterministic error budget.

Coordinated rounding supplies the closest structural foundation. Matrix rounding controls regional errors for two nested or disjoint region families [1]. Two-dimensional sigma–delta quantization controls errors in prefix sums [2]. Its stable greedy scheme uses a finite expanded alphabet whose values may leave the input range. Table 1 makes the combined requirement visible. RangeRectQuant constructs one bounded integer field that supports every future rectangle, preserves designated exact samples, and needs no source at query time. It achieves this combination by characterizing exactly which cumulative errors remain realizable by valid pixels.

Table 1: Guarantees of representative constructions. A checkmark denotes a supplied guarantee; a dash denotes no supplied guarantee. <sup>∗</sup>Uniform pixel bounds require exact reconstruction in the worst case with at least two allowed integers and more changeable pixels than the budget. Clipping and integer rounding do not inherit the original sum budget.
<table><tr><td>Construction</td><td>Regional sums</td><td>Valid integer pixels</td><td>Designated exact values</td></tr><tr><td>Uniform pixel bound</td><td>Exact only*</td><td>V</td><td>V</td></tr><tr><td>Stable sigma-delta</td><td>√</td><td></td><td></td></tr><tr><td>Clip and integer-round</td><td></td><td>√</td><td></td></tr><tr><td>RangeRectQuant</td><td>V</td><td>√</td><td>V</td></tr></table>

## Constructing a Field with Accurate Regional Sums

## Problem definition

Let X be a field of H rows and W columns, with integer samples in $[ 0 , U ]$ , where U is a nonnegative integer. Indices start at one. A mask marks missing samples, represented by zero in both X and the reconstruction Y. The protected set P includes these samples and any designated exact values. For a finite nonnegative budget ε, we require

$$
Y _ { i j } \in \{ 0 , \ldots , U \} , \qquad Y _ { i j } = X _ { i j } \left( ( i , j ) \in P \right) , \qquad \operatorname* { m a x } _ { R } \left| \sum _ { ( i , j ) \in R } ( Y _ { i j } - X _ { i j } ) \right| \leq \varepsilon ,\tag{1}
$$

where R ranges over all nonempty axis-aligned rectangles. Exact transmission of the mask also preserves each region’s valid-sample count. A mean over $k > 0$ valid samples then has error at most $\varepsilon / k$

Let A count unprotected samples and $\delta \geq 0$ be a finite uniform pixel tolerance.

Proposition 1 (Independent pixel bounds). For integer source and output values in $[ 0 , U ]$ , the sharp worst-case rectangular error under independent pixel tolerance $\delta$ is $A \operatorname* { m i n } ( \lfloor \delta \rfloor , U )$ $I f U \geq 1$ and $\varepsilon < A$ , these bounds certify the budget only with exact reconstruction.

Proof. Set $d = \operatorname* { m i n } ( \lfloor \delta \rfloor , U )$ . Each change has magnitude at most d. A zero source and output d at every unprotected sample attain Ad over the whole field. For $\varepsilon < A$ 1， integer d must be zero. □

Define the prefix error $E _ { i j }$ as the sum of $Y - X$ over rows 1 through i and columns 1 through j. Set $E _ { 0 j } = E _ { i 0 } = 0$ , and let $B = \lfloor \varepsilon / 4 \rfloor$ . For $R = [ a , b ] \times [ c , d ]$ 2

$$
\sum _ { R } ( Y - X ) = E _ { b d } - E _ { a - 1 , d } - E _ { b , c - 1 } + E _ { a - 1 , c - 1 } .\tag{2}
$$

Thus $| E _ { i j } | \le B$ implies the regional budget in (1), independently of region size. Our construction keeps this bound while selecting values of Y directly.

Figure 1 shows how coordination changes the possibilities. For $X _ { i j } = 1 , U = 2$ and $P = \varnothing$ , set $Y _ { i j } = 1 - ( - 1 ) ^ { i + j }$ . Every pixel changes. Each rectangle error factors into two alternating sums, each of magnitude at most one. For $H W > 4$ , independent tolerance at least one permits an all-twos field with error HW, so certifying budget four requires exact pixels.

## A row transition that includes the value range

The row becomes a path through prefix errors. The preceding row supplies incoming errors; each new state determines a pixel and must respect its range.

In row $i ,$ the source is $x _ { j } = X _ { i j }$ and the incoming error is $u _ { j } = E _ { i - 1 , j }$ , with $u _ { 0 } = 0$ The previous state is $a = E _ { i , j - 1 } ;$ the new state is $b = E _ { i j }$ . Incoming errors change by $\delta _ { j } = u _ { j } - u _ { j - 1 }$ . Setting $t _ { j } = x _ { j } - \delta _ { j }$ gives reconstructed value $y _ { j } = t _ { j } + b - a$ . The

![](images/4e57cc2eba0850766b5175939c9b5c8bb97a17a4076174f12707a8f4dbb9bcd2.jpg)  
Figure 2: Overview of RangeRectQuant: input X becomes quantized field Y . For integer step $q > 0$ , quotient $Z = \lfloor \bar { Y } / q \rfloor$ and remainder $r = Y - q Z$ use lossless Lempel–Ziv–Markov chain algorithm (LZMA) coding. Decoding restores $Y ;$ prefix tables serve sums and valid counts.

prefix bound requires $b \in [ - B , B ]$ , the range requires $b \in [ a - t _ { j } , a + U - t _ { j } ]$ , and a protected pixel forces $b = a + \delta _ { j }$

We favor values divisible by a positive integer step $q ,$ , called coarse pixels. A fine pixel retains a nonzero remainder modulo q. This transition expresses the coarse condition as

$$
b \equiv a - t _ { j } { \pmod { q } } .\tag{3}
$$

Copying the source row gives a feasible path: $a = u _ { j - 1 }$ and $b = u _ { j }$ produce $y _ { j } = x _ { j }$

## Exact intervals for completing a row

The remaining row may have many legal completions. All prefix-error states that can reach one form a single integer interval.

Only its two endpoints are needed to describe every feasible starting state. Computing them backward lets each forward choice retain a way to finish the row.

Let $V _ { j }$ contain the integer states after column j that allow the remaining row to be completed. Write $\mathbb { Z }$ for the integers and $P _ { i } = \{ j : ( i , j ) \in P \}$ . Starting with $V _ { W } = [ - B , B ] \cap \mathbb { Z }$ , each interval $V _ { j } = [ L _ { j } , H _ { j } ] \cap \mathbb { Z }$ has predecessor

$$
V _ { j - 1 } = \left\{ \stackrel { [ - B , B ] \cap [ L _ { j } + t _ { j } - U , H _ { j } + t _ { j } ] \cap \mathbb { Z } } { [ - B , B ] \cap [ L _ { j } - \delta _ { j } , H _ { j } - \delta _ { j } ] \cap \mathbb { Z } } , \quad j \in P _ { i } . \right.\tag{4}
$$

A backward pass computes these endpoints. During the forward pass, the unprotected pixel’s allowed states are

$$
I _ { j } = V _ { j } \cap [ a - t _ { j } , a + U - t _ { j } ] .\tag{5}
$$

If $I _ { j }$ contains a state satisfying (3), choose the one closest to zero, breaking ties toward the smaller state. Otherwise choose the integer in $I _ { j }$ closest to zero. A protected pixel takes its forced state. Update a after each pixel and u after each row.

Lemma 1 (Exact remaining-row realizability). For preceding-row states $u _ { j } \in [ - B , B ]$ 2 the sets in (4) are exactly those admitting a legal completion. Each is an integer interval, and $0 \in V _ { 0 }$

Proof. An unprotected successor b permits predecessors in $\left[ b + t _ { j } - U , b + t _ { j } \right] \cap \mathbb { Z }$ Their union over consecutive $b \in V _ { j }$ gives (4), including $U = 0$ . Protected transitions translate intervals. Backward induction gives exactness. The legal source sufix supplies $u _ { j } \in V _ { j }$ , including $0 = u _ { 0 } \in V _ { 0 }$ □

Theorem 1 (Linear-time realization of accurate rectangular sums). For every bounded integer input, protected set, integer $q \geq 1$ , and $\begin{array} { r } { f i n i t e \varepsilon \ge 0 } \end{array}$ , the backward–forward construction terminates and satisfies (1). It takes O(HW) arithmetic operations and $O ( W )$ additional row workspace.

Proof. Exact feasibility supplies a successor from each $a \in V _ { j - 1 }$ . Every transition enforces range and protection. Induction gives $| E _ { i j } | \le B$ and (1). Two scans per row use constant-work interval and residue operations with row-length arrays. □

The $H ( H + 1 ) W ( W + 1 ) / 4 = \Theta ( H ^ { 2 } W ^ { 2 } )$ possible rectangles share this guarantee after $O ( H W )$ arithmetic operations. Workspace excludes input, output, and verification tables.

Lemma 2 (Feasible intervals determine fine precision). Write $I _ { j } = [ l , h ] \cap \mathbb { Z }$ at an unprotected forward choice. Its feasible pixel interval is $J _ { j } = [ s , v ] \cap \mathbb { Z }$ , where $s = t _ { j } + l - a$ and $v = t _ { i } + h - a$ . It contains $\kappa _ { j } = \lfloor v / q \rfloor - \lfloor ( s - 1 ) / q \rfloor$ coarse values. Let m count fine pixels in the field, and m<sub>P</sub> count protected source values with nonzero remainder. Then

$$
m = m _ { P } + \sum _ { ( i , j ) \notin P } \mathbf { 1 } \{ \kappa _ { j } = 0 \} \le m _ { P } + \sum _ { ( i , j ) \notin P } \mathbf { 1 } \{ | I _ { j } | < q \} .\tag{6}
$$

Proof. The formula counts multiples of $q$ in $J _ { j }$ , and the forward rule selects one whenever present. Any q consecutive integers contain a multiple. Protected outputs equal their source values. □

Figure 3 maps these intervals and fine pixels in Cairo, with prefix errors bounded throughout.

Encoding the constructed field

Write $Y = q Z + r$ , where $Z = \lfloor Y / q \rfloor$ and $0 \leq r < q$ . LZMA codes both 16-bit arrays after grouping low and high bytes. Files also store masks and metadata.

The intervals determine which pixels can use coarse values. Fine positions and remainders then contribute to encoding length. Set $N = H W$ and let $K = \lfloor U / q \rfloor + 1$ count the possible quotient values; m is the fine count in (6).

Proposition 2 (Geometry-dependent representation length). For $q \geq 2$ and known $N , U , q$ , a lossless representation of Y requires at most

$$
\big \lceil N \log _ { 2 } K \big \rceil + \big \lceil \log _ { 2 } ( N + 1 ) \big \rceil + \bigg \lceil \log _ { 2 } \binom { N } { m } \bigg \rceil + \big \lceil m \log _ { 2 } ( q - 1 ) \big \rceil\tag{7}
$$

bits, plus masks and metadata. For $q = 1 , m = 0$ and the bound is $\lceil N \log _ { 2 } ( U + 1 ) \rceil$

Proof. Encode quotient, count, fine positions, and remainders by ranks among $K ^ { N }$ $N + 1 , \ \binom { N } { m }$ , and $( q \mathrm { ~ - ~ } 1 ) ^ { m }$ possibilities. Each rank uses the ceiling of its set-size logarithm. Decoding forms $Y = q Z + r ;$ at $q = 1$ , only the quotient remains. □

Coarse alphabets reduce quotient cost; fine pixels add positions and remainders.

The band B sets accuracy; q controls storage. For finite nonempty step set Q, let $L _ { q } ( X )$ be the encoded file length at q. Select

$$
\boldsymbol { q } ^ { * } = \arg \operatorname* { m i n } _ { \boldsymbol { q } \in \mathcal { Q } } L _ { \boldsymbol { q } } ( \boldsymbol { X } )\tag{8}
$$

All candidates retain Theorem 1’s guarantee. Ties favor smaller q. Each trial repeats construction and coding.

![](images/75141c650233ed4921c5b9e03c6189239b0721e8b85edd9ae5f1d04a9d756a0c.jpg)

![](images/00366a6e1d1e27643e3daeb223ff0cdde093041d76395d7fb47faf5393daefda.jpg)  
Figure 3: $\mathrm { ~ A ~ } 5 1 2 ^ { 2 }$ Cairo built-area field with 100 m pixels [19], at budget $\varepsilon = 1 0 0 0 \mathrm { { m ^ { 2 } } }$ and step $q = 2 0 0 \mathrm { { m ^ { 2 } } }$ . (a) Source values. (b) Prefix error bounded by $\pm 2 5 0 \mathrm { m } ^ { 2 }$ . (c) Allowed prefix-error states per pixel; darker regions ofer fewer choices. (d) Fine pixels retaining nonzero remainders. All panels share the same grid.

## Evaluation

## Data and file comparisons

We compare self-contained files of two bounded integer rasters: built surface measures building area per pixel; water occurrence records how often water is observed.

Four locations are used to choose the fixed encoding configuration. Four separate cities—Cairo, Delhi, New York, and Jakarta—illustrate the construction and step tradeof. The main built-area rate study uses forty additional locations in six continental groups, selected from metadata before pixel access. Native $1 0 2 4 ^ { 2 }$ windows supply central $5 1 2 ^ { 2 }$ and $2 5 6 ^ { 2 }$ crops. Water uses an independent set of eight metadata-selected $5 1 2 ^ { 2 }$ windows.

Built area uses the 2020 Global Human Settlement Layer: integers in [0, 10000] m<sup>2</sup> on a native 100 m grid [19]. Water occurrence uses the 1984–2024 product [20, 21]: values in [0, 100], valid zero, and missing code 255. Both retain missing samples. Figure 5 shows eight sum budgets per quantity: $2 5 { - } 5 0 0 0 \mathrm { m } ^ { 2 }$ for built area and 5–1000 percentage points for water. The fixed step is $q = \operatorname* { m a x } ( 1 , \lfloor \varepsilon / 5 \rfloor )$ , capped at 100 for water.

A valid self-contained file decodes without the source, preserves the missing mask and protected samples, and meets the integer range and rectangular error budget in (1).

![](images/d5dca5757cf0cb8f6427e9ce5a8987f2e888e4fdb7e3e4aaebde342fafb217c7.jpg)  
1: Stable sigma–delta 2: + clipping 3: + clipping + integer rounding  
4: Projected sigma–delta 5: Prefix rounding 6: + clipping 7: RangeRectQuant  
Figure 4: Regional accuracy and valid pixels. Bars give maximum rectangle error divided by budget; the dashed line marks the target. Crosses mark invalid pixels or over-budget errors. RangeRectQuant meets both requirements.

File sizes include masks, configuration, geospatial metadata, digest, and checksum. We compare six lossless integer representations, lossless JPEG 2000 [22], and SZ3, MGARD, and QPET [4,11,14]. Water adds native-byte LZMA and Portable Network Graphics (PNG) files.

We sweep representative supported configurations of SZ3, MGARD, and QPET. Each decoded field is converted to the required bounded-integer representation and evaluated under the same regional-sum requirement. Files are decoded without access to the source.

The fixed-margin control changes 2 × 2 blocks by (d, −d; −d, d), preserving row and column sums at nine integer bounds. A sum-pyramid control splits square totals among four children within valid ranges. Its files store pixels or deviations of child totals from proportional parent shares.

## Regional accuracy and valid pixels

Regional accuracy and valid pixels must hold together (Figure 4). Stable greedy two-dimensional sigma–delta [2] stores rational outputs exactly. Variants clip, integerround, constrain its alphabet, or round prefixes with optional clipping. Stable sigma– delta controls sums with 2.22–6.76% out-of-range pixels; clipping raises regional error to thousands of times the budget. RangeRectQuant incorporates the range into its choices, delivering both properties in all eight cases.

## Rate across locations and budgets

All 960 proposed built-area files change pixels while preserving the target guarantees. Each is smaller than the smallest valid lossless, scientific-compressor, or fixed-margin file, with aggregate reductions of 16.23–72.17% (Figure 5).

Adding valid sum-pyramid files retains forty wins at budgets 100 and 1000, with savings of 32.55% and 41.30% (Table 2).

Water savings begin at budget 25 and reach 65.35% at 1000, with all eight locations winning from 250. Native-byte lossless files are smaller at 5 and 10.

![](images/b68196f9f0d77a93add9013fe95b8ab5b2a3a0a1b5f15dbaf5322f8c6048cdda.jpg)  
Figure 5: Self-contained rates. (a–c) Forty built-area locations at three sizes. Curves require valid files everywhere; gaps mark incomplete coverage. (d) Eight water locations; budgets use percentage-point sums.

Table 2: File rates in bits per pixel, including masked positions. Reductions compare summed bytes with the smallest valid external files, including 512<sup>2</sup> sum-pyramid controls. Budgets use m<sup>2</sup> for built area and percentage-point sums for water.
<table><tr><td>Quantity and size</td><td>ε</td><td>External</td><td>RangeRectQuant</td><td>Reduction</td><td>Wins</td></tr><tr><td>Built area, 2562</td><td>100</td><td>7.818</td><td>5.421</td><td>30.65%</td><td>40/40</td></tr><tr><td></td><td>1000</td><td>7.398</td><td>3.311</td><td>55.24%</td><td>40/40</td></tr><tr><td>Built area, 5122</td><td>100</td><td>5.878</td><td>3.965</td><td>32.55%</td><td>40/40</td></tr><tr><td></td><td>1000</td><td>4.153</td><td>2.438</td><td>41.30%</td><td>40/40</td></tr><tr><td>Built area, 1024²</td><td>100</td><td>3.646</td><td>2.424</td><td>33.50%</td><td>40/40</td></tr><tr><td rowspan="3">Water, 5122</td><td>1000</td><td>3.535</td><td>1.506</td><td>57.39%</td><td>40/40</td></tr><tr><td>100</td><td>0.486</td><td>0.317</td><td>34.66%</td><td>7/8</td></tr><tr><td>1000</td><td>0.486</td><td>0.168</td><td>65.35%</td><td>8/8</td></tr></table>

## Rate decomposition

Quotient–remainder coding saves 15.20% and 11.03% over direct 16-bit LZMA of identical fields at budgets 100 and 1000, also beating byte-shufled coding (Figure 6). Remainders cost 0.070 and 0.086 bits per pixel; removing them exceeds the budget in all 320 fields.

Step sweeps on the four configuration locations and four separate cities preserve all constraints, with fine choices matching (6). An eight-step probe on the four cities uses common metadata and (8), saving a further 15.04% and 8.30% at budgets 100 and 1000. Search averages 1.20 s per field. Main rate comparisons retain the fixed step.

## Protected samples and scalability

Forty configurations with nested protected sets of up to 10% retain exact samples, pixel range, and regional accuracy. Warm construction with verification takes 22–25 ms.

![](images/7d58bccfcf7c0dcb5ae4a390a79245da904e512e19f88eaa48fa4f005cd5277f.jpg)  
Figure 6: Storage and precision. (a–d) Quotient–remainder file-size reductions versus direct coding of identical $5 1 2 ^ { \hat { 2 } }$ fields at forty alphabetically ordered locations. (e–h) Four separate cities at eight step divisors; $q = \operatorname* { m a x } ( 1$ , ⌊ε/divisor⌋). Dotted lines mark divisor five.

Native $4 0 9 6 ^ { 2 }$ London fields run on a 16 GB Apple M4. Three warmed, singlethread repetitions at budget 100 give medians of 7.84 s encoding and 1.14 s decoding. Construction alone reaches 18.39 million pixels per second.

## Conclusion

Uniform independent pixel bounds can force exact reconstruction. Exact completion intervals make coordinated quantization linear, preserving the pixel range and accuracy of every rectangular sum. Built-surface and water-occurrence studies demonstrate file savings. RangeRectQuant shows how cumulative query structure can guide compression directly: represent future measurements through compact states, characterize which states remain realizable, and quantize within them.

The repository provides code, data, core experiments, and extended figures: https://github.com/Marchematics/RangeRectQuant.

## References

[1] T. Asano, N. Katoh, K. Obokata, and T. Tokuyama, “Matrix rounding under the L<sub>p</sub>-discrepancy measure and its application to digital halftoning,” SIAM J. Comput., vol. 32, no. 6, pp. 1423–1435, 2003.

[2] H. Lyu and R. Wang, “Sigma delta quantization for images,” Commun. Pure Appl. Math., vol. 76, no. 5, pp. 901–945, 2023.

[3] S. Di, J. Liu, K. Zhao, et al., “A survey on error-bounded lossy compression for scientific datasets,” ACM Computing Surveys, vol. 57, no. 11, pp. 287:1–287:38, 2025.

[4] X. Liang, K. Zhao, S. Di, et al., “SZ3: A modular framework for composing predictionbased error-bounded lossy compressors,” IEEE Trans. Big Data, vol. 9, no. 2, pp. 485–498, 2023.

[5] J. Liu, S. Di, K. Zhao, X. Liang, Z. Chen, and F. Cappello, “Dynamic quality metric oriented error bounded lossy compression for scientific datasets,” in Proc. SC, 2022, pp. 1–15.

[6] J. Liu, S. Di, K. Zhao, et al., “High-performance efective scientific error-bounded lossy compression with auto-tuned multi-component interpolation,” Proc. ACM Manag. Data, vol. 2, no. 1, pp. 4:1–4:27, 2024.

[7] Y. Shi, X. Zou, X. Chen, et al., “Machete: An eficient lossy floating-point compressor designed for time series databases,” in Proc. DCC, 2024, pp. 532–541.

[8] T. Lu, Y. Zhong, Z. Sun, et al., “ADT-FSE: A new encoder for SZ,” in Proc. SC, 2023, pp. 1–13.

[9] O. Maireles-Gonz´alez, J. Bartrina-Rapesta, M. Hern´andez-Cabronero, and J. Serra- <sup>\`</sup> Sagrist\`a, “Analysis of lossless compressors applied to integer and floating-point astronomical data,” in Proc. DCC, 2022, pp. 389–398.

[10] F. Cappello, R. Underwood, et al., “What to support when you’re compressing: The state of practice gaps and opportunities for scientific data compression,” in Proc. SC. 2025, pp. 1966–1979, ACM.

[11] M. Ainsworth, O. Tugluk, B. Whitney, and S. Klasky, “Multilevel techniques for compression and reduction of scientific data—the unstructured case,” SIAM J. Sci. Comput., vol. 42, no. 2, pp. A1402–A1427, 2020.

[12] M. Ainsworth, O. Tugluk, B. Whitney, and S. Klasky, “Multilevel techniques for compression and reduction of scientific data—quantitative control of accuracy in derived quantities,” SIAM J. Sci. Comput., vol. 41, no. 4, pp. A2146–A2171, 2019.

[13] P. Jiao, S. Di, H. Guo, et al., “Toward quantity-of-interest preserving lossy compression for scientific data,” Proc. VLDB Endow., vol. 16, no. 4, pp. 697–710, 2022.

[14] J. Liu, P. Jiao, K. Zhao, X. Liang, S. Di, and F. Cappello, “QPET: A versatile and portable quantity-of-interest-preservation framework for error-bounded lossy compression,” Proc. VLDB Endow., vol. 18, no. 8, pp. 2440–2453, 2025.

[15] Y. Liu, S. Di, K. Zhao, et al., “Optimizing error-bounded lossy compression for scientific data with diverse constraints,” IEEE Trans. Parallel Distrib. Syst., vol. 33, no. 12, pp. 4440–4457, 2022.

[16] L. Yan, X. Liang, H. Guo, and B. Wang, “TopoSZ: Preserving topology in error-bounded lossy compression,” IEEE Trans. Vis. Comput. Graph., vol. 30, no. 1, pp. 1302–1312, 2024.

[17] Y. Matias and D. Urieli, “Inner-product based wavelet synopses for range-sum queries,” in Algorithms—ESA 2006. 2006, vol. 4168 of LNCS, pp. 504–515, Springer.

[18] Y. Liu, B. Jiang, T. Yang, S. Di, R. Underwood, and S. Jin, “TOPIQ: Statistical error propagation for quantity-of-interest prediction under lossy compression,” in Proc. SC, 2026, To appear.

[19] M. Pesaresi and P. Politis, “GHS-BUILT-S R2023A: GHS built-up surface grid, derived from Sentinel2 composite and Landsat, multitemporal (1975–2030),” Dataset, 2023.

[20] J.-F. Pekel, A. Cottam, N. Gorelick, and A. S. Belward, “High-resolution mapping of global surface water and its long-term changes,” Nature, vol. 540, no. 7633, pp. 418–422, 2016.

[21] European Commission Joint Research Centre, “Global surface water: Data access and the 1984–2024 update,” 2026, Occurrence version 1.5; accessed October 2, 2026.

[22] M. W. Marcellin, M. J. Gormish, A. Bilgin, and M. P. Boliek, “An overview of JPEG-2000,” in Proc. DCC. 2000, pp. 523–541, IEEE Computer Society.