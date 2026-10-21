# Reusing Assurance Evidence in AI Pipelines: Validity Conditions, a Fail-Closed Release Gate, and a Replay of 714 Public Data Version Control Pipelines

Atanu Mukhopadhyay<sup>a</sup>, Shravya Reddy Nalla<sup>b</sup>

<sup>a</sup>Independent Researcher, New Jersey, USA <sup>b</sup>Independent Researcher, USA

## Abstract

Rerunning every governance control after every change to an artificial intelligence (AI) pipeline is expensive, so teams reuse earlier results. Reuse is safe only if each control’s declared dependencies cover everything that can change its outcome, and only if the2 release gate checks that reused evidence describes the candidate being released. This article makes both conditions explicit and measures how often the first fails in practice. It defines when a signed evidence record is valid for a candidate and gives suficient<sup>2</sup> conditions under which change-local re-execution reaches the same release decision as rerunning every control. A reference gate<sup>t</sup> implementing these checks with Ed25519 digital signatures blocks eight of nine manipulation scenarios, against one for a gate that trusts the planner and three for a gate that only verifies signatures; a dependency the policy never declared passes every gate. A replay of 38,083 commits from 714 public Data Version Control (DVC) pipelines measures that failure mode. Reusing stage results according to declared dependencies avoids 83.4% of stage executions, but in 3.3% of commits it reuses a stage whose imported code or referenced configuration changed, and 31% of pipelines are afected. A manual audit confirms the undeclared dependency in 36 of 40 sampled events. Evaluation stages, whose outputs are the evidence, are the most exposed. Adding each stage’s static import closure lowers stale reuse to 0.4% while still avoiding 81.6% of executions; sampled shadow re-execution removes far less, and rerunning everything after unexplained changes avoids only 45.3%..

Keywords: AI governance, continuous assurance, MLOps, incremental computation, data provenance, mining software repositories<sub>[</sub><sup>c</sup>

## 1. Introduction

An AI release is assembled from data, feature code, training code, configuration, model weights, evaluation sets, and the   
environment that runs them. Governance controls examine that<sub>2</sub> assembly: data validation, performance and fairness evaluation,<sup>.</sup> robustness and privacy tests, security scans, and provenance   
checks. Running all of them after every change is defensible but6   
costly, and much of the cost buys nothing when a change touches2   
only documentation or a component that a control never reads. Pipelines therefore reuse earlier results, sometimes explicitly through stage caches such as DVC [1] and the caching built into ML platforms [2], and sometimes by convention.

Reuse can fail in two diferent ways. The reuse decision can be wrong because the control depends on something its declaration omits, so a cached result describes a computation that no longer exists. Or the evidence presented at release can difer from what it claims to be: produced for an earlier artifact version, copied from another deployment, edited after signing, signed by the team whose work it certifies, or produced under a superseded policy. Recent systems make governance requirements and evidence machine-checkable [3, 4, 5, 6]. To our reading of their published descriptions, none of them defines when a cached result may stand in for a fresh one after the pipeline changes, and no study we are aware of measures how often the dependency declarations of real ML pipelines are complete enough for reuse to be safe.

This article addresses both problems and asks four questions.

RQ1. Which manipulations of reused evidence does a gate that checks evidence validity stop, and which does it not?

RQ2. In real ML pipelines, how much re-execution does dependency-based reuse avoid, and how often does it reuse a result whose undeclared inputs changed?

RQ3. How do alternative invalidation rules, including sampled shadow re-execution, trade avoided work against stale reuse, and what does verification cost at scale?

RQ4. When a stage is reused stale, does executing it again actually produce diferent output?

The contributions are as follows. First, a validity model for signed assurance evidence and suficient conditions under which change-local re-execution is decision-equivalent to exhaustive re-execution (Section 4). The model separates the planner, which only decides what to run, from the gate, which decides what may be released and never consults the planner. Second, a reference implementation of the planner and gate with Ed25519 signatures [7, 8], evaluated against nine manipulation scenarios and two weaker gate designs that are common in practice (Section 5). Third, a replay of 38,083 first-parent commits from 714 public DVC pipelines that compares five invalidation rules against a static reference of each stage’s code and configuration dependencies (Sections 6–7). The replay is checked three ways: a manual audit of sampled events, sandboxed re-execution of flagged stages before and after the change, and a case study of a public-sector valuation pipeline whose maintainers later corrected the same gap. Fourth, an evaluation of sampled shadow re-execution on the real sequence of stale-reuse events, and measurements of planning and verification cost, including parallel throughput.

The study does not claim that static analysis recovers every semantic dependency, and the public repositories it mines are not a sample of industrial pipelines. Its narrower conclusion is that the completeness of dependency declarations, rather than cryptography or gate logic, determines whether evidence reuse is safe, and that inexpensive automatic augmentation of those declarations recovers most of the lost safety.

## 2. Background and related work

## 2.1. Executable governance and assurance evidence

Risk-management frameworks organize AI governance around mapping, measuring, and managing risk [9, 10], and documentation artifacts record evaluation context for models, datasets, and services [11, 12, 13]. Internal auditing frameworks connect these artifacts to an end-to-end review process [14]. Several 2026 systems make the requirements themselves executable. Audit-as-Code maps governance requirements to versioned policy and executable checks over evidence bundles, with risk-tiered gate decisions in CI/CD [3]. Governance-as-Code defines 43 machine-checkable acceptance criteria for EU AI Act obligations, implemented as Rego policies, and validates them on two enterprise deployments [4]. Ontological Knowledge Blocks compile regulatory obligations into SHACL constraints over evidence graphs with PROV-O provenance and evaluate them in an AI-assisted HPC resource-allocation scenario [5]. SACO represents assurance context, its provenance, and explicit information gaps as authoritative semantic structures [6]. These systems establish that evidence can be checked mechanically. The question studied here is diferent: when an existing evidence record may be reused for a new candidate, and how often the inputs that decide that question are declared incompletely.

## 2.2. Provenance and attestation

W3C PROV and OpenLineage describe entities, activities, and derivations for data and pipeline lineage [15, 16]. Software supply-chain systems sign statements about who produced an artifact from which inputs: in-toto chains signed step metadata [17], SLSA grades the trustworthiness of build provenance [18], and Sigstore makes signing and verification routine for open-source artifacts [19]. The evidence records in this article follow the same pattern for governance controls. They bind a measurement to the digests of its inputs, the policy, and the deployment context, and they are signed by an executor that the policy authorizes for that control.

## 2.3. Incremental computation and its correctness

Reusing a result when its inputs are unchanged is the central idea of incremental builds and of regression test selection. Build systems can be characterized by how they decide what to rebuild [20]. Safe regression test selection reruns every test whose dependencies a change touches [21], and dynamic dependency tracking makes this practical at scale [22]. The correctness of both depends on complete dependency information, and that information is often wrong. Unspecified dependencies in makebased builds have been studied empirically [23], build fuzzing uncovered faulty build rules in 31 open-source projects [24], and Rattle avoids declared dependencies altogether by tracing them [25]. This article transfers that question from build outputs to assurance evidence in ML pipelines, where a missed dependency leaves a governance decision resting on a result for a diferent computation.

## 2.4. ML pipelines and pipeline-versioning tools

Production ML depends on automated pipelines, data validation, and tests [26, 27, 28, 29, 30], and asset-management tools track the data, models, and pipelines involved [31]. DVC records pipeline stages in a dvc.yaml file that lists each stage’s command, dependencies, parameters, and outputs, and reruns a stage only when its definition, a declared dependency, or a declared parameter changes [1]. Barrak et al. studied how DVC files co-evolve with source code in 391 GitHub projects and found tight coupling between them [32]. A large-scale study of Jupyter notebooks found that few of them reproduce their stored results [33]. We are not aware of work that measures whether DVC dependency declarations cover what stages actually read, or what that implies for reusing evidence. Table 1 summarizes the positioning.

## 3. Problem and threat model

A candidate is a proposed release. It consists of a set of artifacts identified by content digests, a deployment context (intended use, deployment setting, risk tier), and the digest of the policy it was built under. A control is a check that produces an evidence record: a measurement and a pass or fail status, bound to the digests of the artifacts the control read, the policy digest, the context digest, a validity period, and the identity of the executor that produced it. The policy lists, for each risk tier, the controls that are mandatory, each control’s declared dependencies, its maximum evidence age, and the executor keys allowed to sign its evidence. After a change, a planner selects controls to run, and a gate decides whether the candidate may be released.

The threat model covers faults and misbehavior inside an organization that runs its own pipeline. The planner may be buggy or misconfigured. Teams may present evidence from an earlier artifact version or another deployment, edit stored records, sign their own results, rely on expired evidence, release under a policy that changed during the run, or declare a lower risk tier to shrink the mandatory set. A dependency may simply be missing from a control’s declaration, with no intent to deceive.

Table 1: Positioning relative to the closest lines of work.
<table><tr><td>Line of work</td><td>Checkable evidence</td><td>Reuse validity conditions</td><td>Independent fail-closed gate</td><td>Measures declaration completeness</td><td>Domain</td></tr><tr><td>Executable governance [3, 4, 5, 6]</td><td>Yes</td><td>No</td><td>Yes</td><td>No</td><td>AI governance</td></tr><tr><td>Supply-chain attestation [17, 18, 19]</td><td>Yes</td><td>Partly</td><td>Yes</td><td>No</td><td>Software artifacts</td></tr><tr><td>Build and test dependency studies [23, 24, 25]</td><td>No</td><td>Yes</td><td>No</td><td>Yes</td><td>Builds</td></tr><tr><td>DVC co-evolution [32]</td><td>No</td><td>No</td><td>No</td><td>No</td><td>ML pipelines</td></tr><tr><td>This article</td><td>Yes</td><td>Yes</td><td>Yes</td><td>Yes</td><td>ML pipelines</td></tr></table>

The model assumes that the signing keys of authorized executors are not compromised, that an authorized executor reports what it measured, and that the gate’s code and the active policy are protected. Attacks on those components need the broader supplychain controls cited above and are out of scope.

## 4. Evidence validity model

## 4.1. Validity and admissibility

Let a be a candidate with artifact digests $\nu _ { a } ( \cdot ) ,$ , context $\chi _ { a } ,$ policy digest $\pi _ { a } .$ , and risk tier $r _ { a } .$ . Let P be the active policy with digest $\pi _ { P } ,$ , mandatory sets $M _ { P } ( r )$ , declared dependencies $D _ { P } ( k )$ , maximum ages $\tau _ { P } ( k )$ , and authorized signers $S _ { P } ( k )$ for each control k.

Definition 1 (Valid evidence). An evidence record qfor control k is valid for candidate a at time t if all of the following hold: (i) its signature verifies under a key in $S _ { P } ( k )$ ; (ii) its contract digest equals the digest of k’s contract in P (its dependencies, maximum age, and authorized signers); (iii) its context digest equals the digest of $\chi _ { a } ,$ (iv) its recorded dependency set equals $D _ { P } ( k )$ and, for every $d \in D _ { P } ( k )$ , its recorded digest equals $\nu _ { a } ( d ) ,$ $( \nu ) t \leq q . t _ { \mathrm { e x e c } } + \tau _ { P } ( k ) ,$ and (vi) its status is pass.

Binding a record to its own control’s contract, rather than to the whole policy, means that editing one control’s contract does not invalidate the evidence of the others; admissibility still binds the candidate as a whole to the active policy. Condition (iv) makes validity relative to the declared dependencies. Equality of digests is the default. A control-specific equivalence, for example one that ignores formatting changes in a configuration file, must be part of the policy and is itself evidence that can be reviewed.

Definition 2 (Admissibility). Candidate a is admissible at time $t i f \pi _ { a } = \pi _ { P }$ and, for every $k \in M _ { P } ( r _ { a } )$ , the bundle presented to the gate contains a record validfor a at t.

The risk tier enters the mandatory set, so it cannot be selfdeclared. The prototype treats risk classification as a mandatory control whose evidence binds the context digest. A candidate that claims a lower tier presents a context whose digest no longer matches that record.

## 4.2. Decision equivalence

Let $\Phi _ { k } ( a )$ be the decision of control k on candidate $^ { a , }$ and let $D ^ { * } ( k )$ be the smallest set of inputs (artifacts, configuration, code, environment) whose values can change $\Phi _ { k } . \mathrm { A }$ declaration is sound for k if $D ^ { * } ( k ) \subseteq D _ { P } ( k )$ . Exhaustive evaluation $F ( a )$ runs every mandatory control. Incremental evaluation I(a) reuses a stored record for k whenever it is valid for a and runs k otherwise.

Proposition 1 (Decision equivalence). I(a) and F(a) return the same release decision $i f ( I )$ the declaration is sound for every mandatory control; (2) every value in each $D _ { P } ( k )$ is versioned by digest or compared under an explicit equivalence; (3) each control’s decision is a deterministic function of its declared inputs, or exhaustive and incremental evaluation apply the same statistical decision rule; and (4) the gate implements admissibility.

Proof sketch. Partition the mandatory controls into those with a valid stored record and the rest. The rest are re-executed under both strategies. For a control with a valid record, condition (iv) of validity and condition (2) give equal declared inputs; soundness extends this to every input that can afect the decision, and condition (3) then gives the same decision. Both strategies therefore present the same per-control decisions to the same admissibility predicate. If soundness fails, an input outside $D _ { P } ( k )$ can change $\Phi _ { k }$ while the stored record remains valid, which is a counterexample. □

Proposition 2 (Planner independence). Ifthe gate implements admissibility and derives $M _ { P } ( r _ { a } )$ from the active policy, then no planner output can cause the release ofa candidate that lacks a valid recordfor some mandatory control. Plannerfaults can only cause extra executions or holds.

Proof. Admissibility refers only to the candidate, the active policy, and the presented bundle. The planner’s output does not appear in it. □

Remark 1. Validity is checked against $D _ { P } ( k )$ , not $D ^ { * } ( k )$ . When a declaration is unsound, a record can be valid and still describe a superseded computation, and no gate can tell. The defense is mapping assurance: deriving dependencies automatically where possible, failing closed on changes nobody declared, and sampling reused controls for shadow re-execution. If a fraction $\rho$ of reused controls is re-executed and compared, an unsound declaration that is exercised in n releases escapes detection with probability $( 1 - \rho ) ^ { n }$

## 4.3. Planner

Given a candidate a with predecessor $a ^ { \prime } { \mathrm { . } }$ the planner runs

$$
\begin{array} { r l } & { K _ { \mathrm { e x e c } } = \left( \left\{ k : D _ { P } ( k ) \cap \Delta ( a , a ^ { \prime } ) \neq \emptyset \right\} \right. } \\ & { \quad \quad \quad \cup \left\{ k \in M _ { P } ( r _ { a } ) : \left. \exists q \mathrm { ~ v a l i d } \right\} \cup K _ { \mathrm { t r i g } } \right) \cap M _ { P } ( r _ { a } ) , } \end{array}\tag{1}
$$

where $\Delta ( a , a ^ { \prime } )$ is the set of artifacts whose digests difer and $K _ { \mathrm { t r i g } }$ contains controls triggered by runtime events. If ∆ contains an artifact that no control declares, the planner fails closed and runs every mandatory control. Expired evidence is covered by the second term.

## 5. Reference implementation

The prototype is written in Python 3.14 and uses the cryptography package for Ed25519. It has three parts: signed evidence records, a planner implementing Eq. (1), and a gate implementing admissibility (Figure 1). Records are serialized as canonical JSON (sorted keys, no insignificant whitespace), and each executor holds its own signing key. The policy bundle and each control’s contract are canonicalized and hashed, so every record names the exact contract it was produced under and every candidate the exact policy. The prototype and the analysis scripts were written with the assistance of AI coding tools (OpenAI Codex and Anthropic Claude); the authors specified the designs and analyses, reviewed the code, and checked the reported values against the result files. The gate indexes the presented bundle by control, re-derives the mandatory set from the active policy and the candidate’s context, and returns release or hold with a reason for every unmet requirement. It never reads the planner’s output.

For comparison, the gate can run in two weaker modes that resemble common practice. A planner-trusting gate releases when every control the planner scheduled passed, as in a CI pipeline that promotes when all jobs are green. A signature-only gate verifies that each mandatory control has a signed, passing record and that the candidate names the active policy, but does not check who signed each record, under which contract or context it was produced, or for which artifact versions.

## 6. Evaluation design

## 6.1. RQ1: manipulation scenarios

Each scenario starts from a release with complete, valid evidence, applies a change, and then applies one manipulation (Table 3). The policy has nine controls, three risk tiers, and one authorized executor per control. A separate model-team key is registered in the keyring but not authorized for any assurance control. Two scenarios are legitimate reuse and must release; one is an honest failure and must hold; nine are manipulations or faults that must hold.

Table 2: Corpus construction.
<table><tr><td>Step</td><td>Repositories</td></tr><tr><td>Candidates from search</td><td>3,524</td></tr><tr><td>Root dvc . yaml on default branch</td><td>2,130</td></tr><tr><td>Cloned successfully</td><td>2,087</td></tr><tr><td>At least two stages at HEAD</td><td>1,880</td></tr><tr><td>At least one resolvable Python stage</td><td>1,644</td></tr><tr><td>At least 10 analyzable commits</td><td>726</td></tr><tr><td>One repository per root commit</td><td>714</td></tr><tr><td>Commits analyzed</td><td>38,083</td></tr></table>

## 6.2. RQ2–RQ3: replay of public DVC pipelines

Corpus. On 13 October 2026 we collected candidate repositories from two sources: the results of a Sourcegraph code search for files named dvc.yaml, and GitHub repository searches for DVC- and MLOps-related topics and README text. Forks and archived repositories were excluded. Of 3,524 candidates, 2,130 had a dvc.yaml at the root of the default branch and 2,087 could be cloned. The filters in Table 2 were fixed before any result was examined. Repositories sharing a root commit are copies of the same history, and only the copy with the most analyzable commits was kept. The final corpus has 714 repositories and 38,083 commits; the median repository contributes 23 commits and defines 4 stages.

Replay. Each pipeline stage stands for a control whose evidence is its output. For every first-parent commit after dvc.yaml first appears, the replay compares the commit with its parent and decides which stages must re-execute. It emulates DVC’s own rule. A stage is invalidated if a declared dependency changed (a file, a directory prefix, or a path templated from parameters), if a declared parameter key changed value, or if the stage definition changed after resolving $\$ \downarrow . . \}$ templates against params.yaml. Invalidation propagates to every stage that consumes an invalidated stage’s outputs, and a change to a .dvc pointer counts as a change to the data it tracks.

Reference dependencies. For each stage, static analysis resolves the entry points in its command: python script.py, python -m module, and script or notebook files named on the command line. It then computes the transitive closure of local Python imports, including package \_\_init\_\_ modules and relative imports. String literals in that closure that name configuration files in the repository (YAML, JSON, TOML, INI) are added; a change to such a file counts only if a changed leaf key is named somewhere in the stage’s code. A change to Python code counts only if the parsed syntax tree changes after docstrings are removed, so comment and formatting edits are ignored, and a change to a notebook counts only if a code cell changes. Changes to environment files (requirements, lock files, pyproject.toml, Dockerfiles) are reported separately, because they plausibly afect every stage but their relevance cannot be decided statically. The reference set is a lower bound on true dependencies. It misses dynamic imports, files opened through computed paths, and data read outside the repository.

![](images/8886ff28b76c46cc40c147a8046d696050092e5cbf4c24043a748a98c7db57e9.jpg)  
Figure 1: Separation of the planner and the gate. The planner decides which controls to run and may reuse stored evidence. The gate re-derives the mandatory set from the active policy and checks every presented record for signer authorization, policy and context binding, dependency digests, freshness, and status.

Invalidation rules. Five rules are compared. Full reruns every stage after every commit. Declared is DVC’s rule. Declared+imports adds each stage’s import closure to its declared dependencies. Declared+imports+environment also reruns every Python stage when an environment file changes. Fail-closed applies the declared rule but reruns every stage when a changed code, configuration, notebook, or environment file is declared by no stage, which is the conservative fallback of Eq. (1). For each rule we report the share of stage slots executed and the share of commits in which at least one stage that the reference marks as invalidated is reused (a stale reuse). Intervals are 95% percentile intervals from 2,000 bootstrap resamples of repositories.

Stage roles and change types. To see where evidence goes stale, each stage was assigned a role from its name and command: evaluation (names containing eval, test, metric, score, benchmark, assess, report, or compare), training (train, fit, tune, hyperparameter search, fine-tuning), data preparation (ingest, load, download, prepare, clean, split, feature, transform, embed, index, validate), or other. Commits were grouped by the categories of files they changed.

Audit. From the commits in which the declared rule reuses an invalidated stage, we drew a random sample of 40 and examined each flagged stage, the import chain from its entry point to the changed file, and the dif. Each event was labeled material (a real undeclared dependency whose change plausibly alters the stage’s output or measurement), benign (a real undeclared dependency whose change does not, for example a new unused constant or a type annotation), or analyzer error (no real dependency, or DVC would in fact rerun the stage). An earlier audit round on a separate sample was used only to correct the analyzer before the final run. After the main results, we also measured declaration hygiene at each repository’s HEAD (whether each stage declares its entry script and its local imports) and recomputed the main metrics on corpus subsets. These analyses were not pre-specified.

Shadow re-execution. Shadow runs re-execute a sample of reused stages and compare their outputs with the cached ones. We replayed each repository’s history in first-parent order. At every commit, each stage that the declared rule reused was shadow-executed with probability $\rho \in \{ 1 , 5 , 1 0 , 2 0 \} \%$ . A shadow run of a stale stage exposes the problem with probability q: q = 1 if any stale input changes the output, or $q = 0 . 3 5$ , the audited share of material changes. Two responses to an exposure were compared: declaring the dependency that was missed, or switching the repository to declarations augmented with each stage’s import closure. Each configuration was repeated 200 times.

## 6.3. RQ3: cost

Gate verification time was measured for 9 to 1,000 mandatory controls (200 repetitions each). The full prototype was also run over the commit histories of the corpus (up to 400 commits per repository): each stage becomes a control whose dependencies are its declared dependency paths, parameter keys, and upstream outputs; at each commit the planner selects stages, the selected stages’ executors sign fresh records, and the gate admits the candidate from the merged bundle. Parallel throughput was measured by admitting 2,000 candidates, each with 50 signed records, with one gate per worker process and candidates divided among 1 to 12 processes. The planner core was also benchmarked on synthetic sparse graphs of up to 500,000 nodes. All timings were taken on one Apple Silicon laptop (14 logical CPUs) running Python 3.14.

## 6.4. RQ4: re-execution of flagged stages

The audit judges whether a change could alter a stage’s output; re-execution measures it. For commits in which the declared rule reused a stage that the reference marks as invalidated, the stage was executed twice at the parent commit and once at the flagged commit, and its declared outputs were compared. If the two parent runs agree and the commit run difers, the reused output was not what a fresh run produces. A flagged stage was eligible if it and every stage upstream of it (at most six stages) run a single Python entry point, every input that no stage in that chain produces is a file in the repository (no data held in a DVC remote), the code does not contact the network, and the environment names no GPU-only packages. Upstream stages were executed in order before the flagged stage, and at most three events were taken per repository. Each repository’s environment was installed with uv into its own Python 3.11 environment from requirements.txt or pyproject.toml, together with a small common base (NumPy, pandas, scikit-learn, SciPy, PyYAML, Matplotlib, seaborn, joblib). If the requirements could not be resolved together, each line was installed on its own; if a run failed on a missing module, that module was installed and the run retried, at most twice. Because the code comes from unvetted public repositories, every stage run executed under the macOS sandbox with networking disabled, reads of the user’s home directory denied, and writes confined to a scratch directory, with a three-minute limit per run. Dependency installation needs the network and ran outside the sandbox. The protocol was revised once. A first version, which required a requirements file, ran no upstream stages, and allowed no deep-learning frameworks, completed too few events to be informative; we report its outcome alongside the final one.

## 7. Results

## 7.1. RQ1: which manipulations the gate stops

Table 3 reports the decisions. The full gate released both legitimate candidates, held the honest failure, and held eight of the nine manipulations. The planner-trusting gate held only expired evidence, because a correct planner re-ran the expired control and the fresh run failed. The signature-only gate also held tampered records, missing evidence, and candidates built under a superseded policy. Every gate released the undeclareddependency scenario (S10): the stored fairness record was valid for the declared inputs, and the cohort definition that changed was not among them. This is the limit stated in the remark of Section 4, and it is the failure mode that the replay measures.

## 7.2. RQ2: reuse and stale reuse in real pipelines

The commits span the usual work of a pipeline project: 34% touched code, 21% configuration or parameter files, 11% DVC data pointers, 12% environment files, and 7% the pipeline definition itself, while 18% changed only documentation. Under the declared rule, 71% of commits re-executed no stage at all.

![](images/ca2079da3b00eae0f3295dc891d64ab9f12251b174387c7781bd6852387bc51f.jpg)  
Figure 2: Avoided stage executions against commits with stale reuse, for each invalidation rule. Error bars: 95% bootstrap intervals over repositories.

Table 5 and Figure 2 compare the five rules. Under DVC’s declared rule, pipelines executed 16.6% of stage slots, so 83.4% of stage results were reused. In 3.3% of commits (95% interval 2.4–4.5%), at least one reused stage had an imported module, an entry script, or a referenced configuration key that changed. Counting environment-file changes as well raises the share to 13.2%. Stale reuse is concentrated: 218 of the 714 pipelines (31%) were afected at least once, and the median pipeline was never afected. Among stages that record metrics or plots, the outputs most likely to be used as evaluation evidence, the declared rule reused a stale stage in 0.9% of commits. Code dependencies (an imported module or the stage’s own entry script) caused 88% of the flagged commits; configuration keys read through a string reference caused the rest.

The declarations themselves explain why. At each repository’s HEAD, 54% of the 3,969 stages with a Python entry point had at least one undeclared local code dependency, and 469 of 711 pipelines (66%) contained such a stage. 6% of stages did not declare their own entry script, and 74% of the stages that import local modules left at least one of them undeclared. Most of these gaps are harmless most of the time, because the undeclared file rarely changes without a declared one; the replay measures how often it does. Table 4 shows that the result is not driven by a few large repositories or by abandoned projects. Across the subsets, the declared rule reused a stale stage in 3.0–4.8% of commits, and the import-closure rule in 0.1–0.5%.

The audit (Table 6) confirmed a real undeclared dependency in 36 of 40 sampled events (90%; Wilson 95% interval 77–96%). In 14 events (35%; 22–50%), the change plausibly altered the reused output. Examples include a rewritten system prompt and a changed generation configuration for LLM agents under benchmark, an evaluation metric switched from pmin to pmax in an R script that its stage did not declare, a text-cleaning step re-enabled in an undeclared transformation module, a diferent model constructor in a training module, and a changed learningrate warmup schedule. The benign events were dominated by added helper functions and constants, logging and experimenttracking changes, and refactorings that preserve behavior. All four analyzer errors were metrics files written by another stage, which a shared script named as a string but did not read. Combining the audit with the replay, roughly 1.2% of commits reused a stage whose change plausibly altered its output.

Table 3: Gate decisions in the manipulation scenarios. R = release, H = hold. Correct decisions are marked with an asterisk.
<table><tr><td></td><td>Scenario</td><td>Planner-trusting</td><td>Signature-only</td><td>Full gate</td></tr><tr><td>SO</td><td>Documentation-only change. Only provenance is invalidated and re-run; other evidence is reused.</td><td>R*</td><td>R*</td><td>R*</td></tr><tr><td>S1</td><td>Dataset change, correct re-run. Invalidated controls re-run and pass.</td><td>R*</td><td>R*</td><td>R*</td></tr><tr><td>S2</td><td>Planner omits an invalidated control. Fairness evidence for the old dataset is presented.</td><td>R</td><td>R</td><td>H*</td></tr><tr><td>S3</td><td>Cross-context replay. Evidence certified for a different deployment is reused.</td><td>R</td><td>R</td><td>H*</td></tr><tr><td>S4</td><td>Tampered evidence record. A failing privacy result is edited to pass after signing.</td><td>R</td><td>H*</td><td>H*</td></tr><tr><td>S5</td><td>Self-certified result. The model team signs its own robustness pass.</td><td>R</td><td>R</td><td>H*</td></tr><tr><td>S6</td><td>Expired evidence. Evidence older than its maximum age is presented.</td><td>H*</td><td>R</td><td>H*</td></tr><tr><td>S7</td><td>Policy changed before release. The candidate was planned and evaluated under a policy that has since changed.</td><td>R</td><td>H*</td><td>H*</td></tr><tr><td>S8</td><td>Risk-tier downgrade. A tier-3 system is submitted as tier 1.</td><td>R</td><td>R</td><td>H*</td></tr><tr><td>S9</td><td>Mandatory control never run. Explainability evidence is absent.</td><td>R</td><td>H*</td><td>H*</td></tr><tr><td>S10</td><td>Undeclared semantic dependency. Cohort definition changes but is absent from fairness dependencies.</td><td>R</td><td>R</td><td>R</td></tr><tr><td>S11</td><td>Control fails honestly. A re-run fairness control fails.</td><td>H*</td><td>H*</td><td>H*</td></tr><tr><td colspan="2">Manipulations held (S2-S10)</td><td>1/9</td><td>3/9</td><td>8/9</td></tr></table>

Table 4: Sensitivity of the replay results to the corpus. Executed and stale-reuse shares as in Table 5; brackets are 95% bootstrap intervals over repositories.
<table><tr><td>Subset</td><td>Repositories</td><td>Commits</td><td>Declared: executed (%)</td><td>Declared: stale reuse (%)</td><td>Declared + imports: stale reuse (%)</td></tr><tr><td>All repositories</td><td>714</td><td>38,083</td><td>16.6</td><td>3.3 [2.4, 4.5]</td><td>0.4</td></tr><tr><td>At least 50 analyzed commits</td><td>145</td><td>25,992</td><td>16.6</td><td>3.6 [2.3, 5.4]</td><td>0.4</td></tr><tr><td>At least 5 stages at HEAD</td><td>330</td><td>19,355</td><td>13.4</td><td>4.8 [3.3, 6.7]</td><td>0.5</td></tr><tr><td>At least 5 GitHub stars</td><td>22</td><td>5,343</td><td>46.4</td><td>3.0 [0.6, 10.3]</td><td>0.1</td></tr><tr><td>Last commit in 2026</td><td>548</td><td>28,696</td><td>12.5</td><td>3.0 [2.0, 4.2]</td><td>0.3</td></tr><tr><td>Largest 1% of repositories removed</td><td>707</td><td>27,768</td><td>15.3</td><td>4.2 [3.2, 5.2]</td><td>0.5</td></tr><tr><td>Repository-weighted mean</td><td>714</td><td>38,083</td><td>18.0</td><td>3.2</td><td>0.5</td></tr></table>

Table 7 shows where stale reuse occurs. Evaluation stages, whose outputs are the measurements that governance decisions rest on, were the most exposed: 4.9% of reused evaluation stages were stale, against 2.0% of data-preparation stages, 2.8% of training stages, and 1.7% of other stages. Evaluation stages held 15% of stage slots but accounted for 29% of stale stage reuses. By change type, 8.9% of commits that touched code led to stale reuse, against 5.7% of commits that touched configuration and 0.9% of commits that changed only DVC data pointers; no commit that changed only documentation did.

## 7.3. A case: the Cook County residential valuation pipeline

The Cook County Assessor’s Ofice publishes the pipeline behind its residential automated valuation model, which produces initial assessed values for single- and multi-family residential properties in Cook County, Illinois [34]. Its dvc.yaml declares data inputs, parameters, and outputs for each stage but, until August 2024, not the R scripts that the stages run. The project documentation tells users to run the pipeline with dvc repro, noting that each stage runs only if necessary. On 24 January 2024 a commit titled “Boot 0 or sub-0 values from ratio stats” changed one line of pipeline/03-evaluate.R, replacing pmin with pmax in the bound applied to estimated values before the assessor-specific ratio statistics (COD, PRD, and PRB) are computed, so that estimates are floored at 1 rather than capped at 1. These statistics measure the uniformity and vertical equity of assessments, and the evaluation stage reports them for every geography and property class. Because the script was not a declared dependency, dvc repro would have reused the previous evaluation outputs, and a gate checking declared dependencies would have accepted them as valid. Over the 264 commits before the fix, the replay found stale reuse in 83 (31%), including 50 stale reuses of the evaluation stage. In August 2024 the maintainers added each stage’s script to its dependencies (“Edit DVC deps to include cmd run files”), and the replay found no stale reuse in the 141 commits that followed. The case illustrates the failure mode in a maintained production pipeline. It also gives an independent check on the analyzer: the change the maintainers made is the one the analysis recommends, and it removed every flagged event.

Table 5: Invalidation rules on 38,083 commits from 714 DVC pipelines. Executed: share of stage slots re-executed. Stale reuse: share of commits in which at least one stage invalidated under the reference is reused, counting code and configuration dependencies (environment changes in the last column). Brackets: 95% bootstrap intervals over repositories.
<table><tr><td>Rule</td><td>Executed (%)</td><td>Stale reuse, code/config (%)</td><td>Stale reuse, incl. environment (%)</td></tr><tr><td>Full</td><td>100.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Declared (DVC)</td><td>16.6 [11.4, 25.0]</td><td>3.3 [2.4, 4.5]</td><td>13.2</td></tr><tr><td>Declared + imports</td><td>18.4 [12.8, 27.0]</td><td>0.4 [0.3, 0.7]</td><td>10.5</td></tr><tr><td>Declared + imports + environment</td><td>26.2 [19.8, 35.3]</td><td>0.4 [0.2, 0.6]</td><td>0.4</td></tr><tr><td>Fail-closed fallback</td><td>54.7 [48.6, 62.2]</td><td>0.4 [0.2, 0.6]</td><td>0.4</td></tr><tr><td>Reference (code/config)</td><td>18.7</td><td>0.0</td><td></td></tr></table>

Table 6: Manual audit of 40 randomly sampled events in which the declared rule reused an invalidated stage.
<table><tr><td>Label</td><td>Events</td><td>Share (%)</td></tr><tr><td>Material: real dependency, output plausibly changed</td><td>14</td><td>35</td></tr><tr><td>Benign: real dependency, behavior-neutral change</td><td>22</td><td>55</td></tr><tr><td>Analyzer error</td><td>4</td><td>10</td></tr></table>

## 7.4. RQ3: trading avoided work against stale reuse

Adding each stage’s import closure to its declaration (Declared + imports) lowered the share of commits with stale code or configuration reuse from 3.3% to 0.4% and raised executed stage slots only from 16.6% to 18.4%. The fail-closed fallback reached 0.4%, close to the import-closure rule, but executed 54.7% of stage slots, because many commits change some code or configuration file that no stage declares. Treating environment changes as dependencies of every Python stage closed most of the remaining environment exposure, at 26.2% of slots executed. The residual stale reuse under the augmented rules comes from configuration keys read by string reference, which the import closure does not add.

Shadow re-execution is a weaker substitute (Figure 3). Reexecuting 10% of reused stages and declaring each dependency that a shadow run exposes reduced stale-reuse commits by only 22% when every stale input changes the output, and by 12% when only the audited share of material changes is detectable. Undeclared dependencies keep arriving through new files, so fixing them one at a time lags behind. Using the first exposure as a signal to switch the repository to import-closure declarations did better, reducing stale reuse by 55% (41% under the conservative assumption), but still less than adopting the import closure from the start (87%). At 10% sampling, 46% of afected repositories had at least one exposure during their history. Shadow runs are best used to detect that declarations are unsound, not to compensate for it.

Verification is inexpensive relative to the controls it protects (Figure 4). Gate time grew linearly at about 42 µs per record, dominated by Ed25519 verification: the 95th percentile was 0.40 ms for nine mandatory controls and 45 ms for 1,000. The planner selected controls in under 1 ms at 1,000 controls. Replaying the corpus histories through the full prototype covered

Table 7: Stale reuse under the declared rule by stage role (top) and by the type of file a commit changed (bottom).
<table><tr><td>Stage role</td><td>Stage slots</td><td>Stale (% of reused)</td><td>Share of all stale (%)</td></tr><tr><td>Data preparation</td><td>75,793</td><td>2.0</td><td>32</td></tr><tr><td>Training</td><td>33,689</td><td>2.8</td><td>18</td></tr><tr><td>Evaluation</td><td>31,043</td><td>4.9</td><td>29</td></tr><tr><td>Other</td><td>68,342</td><td>1.7</td><td>21</td></tr><tr><td>Commit touched</td><td>Commits</td><td>Stale reuse (% of commits)</td><td></td></tr><tr><td>Code</td><td>13,047</td><td>8.9</td><td></td></tr><tr><td>Configuration or parame-</td><td>8,168</td><td>5.7</td><td></td></tr><tr><td>ters Notebook</td><td>1,814</td><td>5.7</td><td></td></tr><tr><td>Tests</td><td>4,683</td><td>5.9</td><td></td></tr><tr><td>Pipeline definition</td><td>2,557</td><td>6.1</td><td></td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>Environment files</td><td>4,579</td><td>4.0</td><td></td></tr><tr><td>CI configuration</td><td>3,864</td><td>1.8</td><td></td></tr><tr><td>DVC data pointer</td><td>4,122</td><td>0.9</td><td></td></tr><tr><td>Documentation only</td><td>6,930</td><td>0.0</td><td></td></tr></table>

30,998 commits from 714 pipelines. The 95th-percentile time to plan, sign, and admit a candidate was 44.3 ms, most of it spent hashing declared directory dependencies; the gate alone took 0.75 ms at the 95th percentile, and the slowest commit, in a pipeline with 150 stages, took 5.2 s. The prototype’s planner selected the same number of stages as the replay’s declared rule in 97% of commits, and the gate released all 1,237 commits in which the replay found stale reuse. This confirms on real histories that a gate checking declared dependencies cannot see an undeclared one. Verification parallelizes across candidates: one process verified 21,849 records per second and twelve processes 179,010, or 3,580 candidates with 50 controls each per second. The planner core processed sparse graphs of 500,000 nodes in 9.9 ms at the 95th percentile, with 97 MiB of peak traced memory.

## 7.5. RQ4: re-execution of flagged stages

Of the 124 flagged events in 39 repositories that met the eligibility conditions, 40 were sampled. Three events, from three repositories, completed all three runs with agreeing parent runs. In one of them the stage’s output at the flagged commit difered from its output at the parent commit, and in two it did not. In one further event the stage ran at the parent commit but failed at the flagged commit, so reuse would have certified a stage that no longer ran. Four events gave diferent outputs in the two parent runs and were set aside. The remaining events could not be reproduced ofline from the repository: a file that no declared dependency provides was missing (5); a package the environment did not declare was missing (5); the code raised another error (11); the code tried to reach the network (2); the requirements could not be installed (6); the stage ran but wrote none of its declared outputs (1); the checkout could not be exported (2). Missing inputs and missing packages are incomplete declarations in their own right: a stage whose data dependencies and environment were fully declared would run from a checkout. A first, stricter version of the harness, which required a requirements file and did not run upstream stages, completed one of 22 sampled events; its output changed.

![](images/22577dec7a73272394e9b35c91b56ed1d4b0ccd8c29981f31c73397d67aab795.jpg)

Figure 3: Reduction in commits with stale reuse when a share of reused stages is re-executed as shadow runs, for two responses to an exposure and two exposure probabilities q (200 repetitions per point). The dotted line is the reduction from adopting import-closure declarations at the start.  
![](images/e47c5f47d8af46fa61d324177bee7bd69394e6ced66c1a0714f1441c00058c57.jpg)  
Figure 4: Gate verification and planning time against the number of mandatory controls (95th percentile of 200 repetitions).

The event whose output changed is a churn-prediction pipeline. Its dataset stage is invoked as python src/data/ load\_data.py --config=params.yaml, but params.yaml is neither a declared dependency nor a declared parameter. The flagged commit added 13 features to the feature list in that file, and the stage’s output, the raw training table, changed accordingly; under dvc repro the old table would have been reused. One of the two identical outcomes was flagged through a metrics file that another stage writes, which is the pattern the audit classified as an analyzer error, so its identical output is consistent with that classification. The re-execution results are few, and they mostly show how rarely a public pipeline can be re-executed from what it declares. Where re-execution was possible, it confirmed both kinds of event the audit described: stale reuse that changes the evidence, and stale reuse that does not.

## 8. Discussion

## 8.1. What determines whether reuse is safe

The scenarios and the replay point to the same conclusion from opposite directions. A gate that checks signer authorization, policy and context binding, dependency digests, and freshness stops the manipulations that weaker gates miss, and it does so in milliseconds. It cannot stop reuse that the declarations themselves permit. In the public pipelines, declarations were incomplete often enough to matter: roughly one commit in thirty reused a stage whose code or configuration had changed, and nearly one pipeline in three was afected at least once. Most of these changes were behavior-neutral, which explains why incomplete declarations persist unnoticed. A minority changed what the reused stage computes, and for evaluation stages that means a release decision resting on a measurement of something else.

Three practices follow. First, derive code dependencies automatically. The import closure is cheap to compute and removed most stale reuse while costing about 1.8 percentage points of executions. Second, treat the environment as a dependency. Requirements and lock files changed in 12% of commits, and their efect cannot be ruled out statically; pinning environments by image or lock-file digest makes this dependency explicit. Third, sample reused controls for shadow re-execution, but treat a discrepancy as a signal that the declarations are unsound rather than as a single dependency to patch. The replay shows why: at a 10% sampling rate, patching each exposed dependency removed 22% of stale-reuse commits, while switching to automatic declarations after the first exposure removed 55%. Because most stale reuse is benign, comparing shadow outputs, rather than inputs, also separates the material cases from the harmless ones.

## 8.2. Scalability and eficiency

Reuse avoided most stage executions in every rule except the fail-closed fallback, and the augmented declarations kept that saving while removing most of the risk. The avoided executions are the eficiency gain. The public repositories do not record stage runtimes or energy, so the gain is reported as executions rather than energy. Verification adds tens of microseconds per record and planning remains sub-millisecond for policies of hundreds of controls, so neither limits release frequency. At organizational scale the expensive part is maintaining sound declarations, which is why automatic derivation matters more than faster verification.

## 8.3. Running the gate at scale

The gate is stateless apart from the policy and the keyring, so it can be replicated and candidates can be sharded across replicas by pipeline. The measured throughput of one laptop process exceeds the commit rate of the whole corpus by orders of magnitude, so verification cost is not the obstacle to deployment. Three integration points matter more. First, evidence records map directly onto in-toto attestation statements [17]: the artifact digests are the subject, and the measurement, decision rule, contract digest, and context digest form the predicate. Signing them through Sigstore [19] with the executor’s workload identity, and logging them in a transparency log, gives the append-only store that the threat model assumes. Second, declared dependencies should not be written by hand. Lineage events such as OpenLineage run facets [16] record the inputs that a job actually read, and combined with the static import closure studied here they can populate a control’s contract automatically. Third, on shared clusters the gate fits naturally at admission time, before a release job is scheduled, so that a candidate with stale evidence is held without consuming compute. In HPC and data-center settings the saving that matters is the evaluation work avoided, and the replay shows that most of it can be kept while removing most stale reuse.

## 8.4. Threats to validity

Construct validity. The reference dependency set is produced by static analysis and is itself incomplete. It misses dynamic imports, computed file paths, and data outside the repository, so the reported stale-reuse rates are lower bounds. It can also over-approximate, because importing a module does not mean using the part that changed; the audit measures this, and the benign category is large. Internal validity. The analyzer emulates DVC’s invalidation rule rather than running DVC, and an earlier audit round found two behaviors that needed emulating (template resolution and Windows path separators) before the final run. External validity. Public DVC repositories include many course and personal projects [35]. Commits are not release candidates, and industrial pipelines may declare dependencies more carefully or less. Results are reported per commit and per repository to show how concentrated the efect is. Re-execution. Only stages that run ofline from a plain checkout could be reexecuted, which favors small projects with data in the repository; the re-execution results describe that subset. Prototype. The gate is not integrated with an orchestrator, model registry, or transparency log, and its scenarios test logic under the stated threat model rather than resistance to a compromised signer.

## 9. Conclusion

Reusing assurance evidence is safe under conditions that can be stated precisely. A gate can enforce all of them except one, the soundness of dependency declarations, and that one is violated in real ML pipelines often enough to matter. In 714 public DVC pipelines, declared-dependency reuse avoided 83.4% of stage executions and reused a stale stage in 3.3% of commits. Adding static import closures removed most of that exposure while keeping most of the saving. Evaluation stages, whose outputs are the evidence itself, were the most exposed, and a public-sector valuation pipeline showed the same gap until its maintainers closed it in the way the analysis recommends. Governance platforms that reuse evidence should derive dependencies automatically, treat environments as dependencies, use sampled re-execution to detect unsound declarations rather than to repair them one at a time, and keep the gate independent of the planner. Replaying industrial pipelines with measured stage runtimes, and tracing dependencies dynamically rather than statically, are the most useful next steps.

## CRediT authorship contribution statement

Atanu Mukhopadhyay: Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Visualization, Writing – original draft, Writing – review & editing. Shravya Reddy Nalla: Software, Formal analysis, Validation, Writing – review & editing.

## Declaration of competing interest

The authors declare no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Funding

This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors.

## Data availability

The analysis code, the prototype, the list of repositories with the commit identifiers analyzed, the audit labels, and the summary results are provided as ancillary files with this preprint. The complete per-commit and per-stage results are available from the authors. The mined repositories are public; their contents are not redistributed.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors utilized OpenAI Codex and Anthropic Claude to check English grammar, spelling, and references. After employing these tools, the authors reviewed and edited the content as needed, taking full responsibility for the published article’s final content.

## References

[1] R. Kuprieiev, et al., DVC: Data Version Control - Git for Data & Models, Zenodo (2026). doi:10.5281/zenodo .3677553.

[2] D. Baylor, E. Breck, H.-T. Cheng, N. Fiedel, C. Y. Foo, Z. Haque, S. Haykal, M. Ispir, V. Jain, L. Koc, C. Y. Koo, L. Lew, C. Mewald, A. N. Modi, N. Polyzotis, S. Ramesh, S. Roy, S. E. Whang, M. Wicke, J. Wilkiewicz, X. Zhang, M. Zinkevich, TFX: A TensorFlow-Based Production-Scale Machine Learning Platform, in: Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 2017, pp. 1387–1395. doi:10.1145/3097983.3098021.

[3] A. E. Muhammad, K.-C. Yow, S. Alsenan, Audit-as-code: a policy-as-code framework for continuous AI assurance, Front. Artif. Intell. 9 (2026) 1759211. doi:10.3389/fr ai.2026.1759211.

[4] R. K. Paul, S. Nandy, Governance-as-Code: Translating EU AI Act Technical Requirements into Executable Compliance Pipelines for Generative AI Systems, arXiv preprint arXiv:2609.20016 (2026). doi:10.48550/arXiv.2609. 20016.

[5] A. K. Sharma, J. M. Kunkel, Ontological Knowledge Blocks: Executable Compliance and Profile-Based Validation for Trustworthy AI Systems, arXiv preprint arXiv:2605.23297 (2026). doi:10.48550/arXiv.2 605.23297.

[6] S.-F. Wen, Ontological Foundations for Deterministic Assurance Context Construction and Governed AI Reasoning, Appl. Sci. 16 (4) (2026) 1984. doi:10.3390/app16041 984.

[7] D. J. Bernstein, N. Duif, T. Lange, P. Schwabe, B.-Y. Yang, High-speed high-security signatures, J. Cryptogr. Eng. 2 (2) (2012) 77–89. doi:10.1007/s13389-012-0027-1.

[8] S. Josefsson, I. Liusvaara, Edwards-Curve Digital Signature Algorithm (EdDSA), RFC 8032, Internet Engineering Task Force (2017). doi:10.17487/RFC8032.

[9] National Institute of Standards and Technology, Artificial Intelligence Risk Management Framework (AI RMF 1.0), Tech. Rep. NIST AI 100-1, National Institute of Standards and Technology (2023). doi:10.6028/NIST.AI.100-1.

[10] International Organization for Standardization, ISO/IEC 42001:2023 Information Technology—Artificial Intelligence—Management System, International standard (2023).

[11] M. Mitchell, S. Wu, A. Zaldivar, P. Barnes, L. Vasserman, B. Hutchinson, E. Spitzer, I. D. Raji, T. Gebru, Model Cards for Model Reporting, in: Proceedings of the Conference on Fairness, Accountability, and Transparency, 2019, pp. 220–229. doi:10.1145/3287560.3287596.

[12] T. Gebru, J. Morgenstern, B. Vecchione, J. W. Vaughan, H. Wallach, H. Daumé III, K. Crawford, Datasheets for datasets, Commun. ACM 64 (12) (2021) 86–92. doi: 10.1145/3458723.

[13] M. Arnold, R. K. E. Bellamy, M. Hind, S. Houde, S. Mehta, A. Mojsilovic, R. Nair, K. N. Ramamurthy, A. Olteanu,´ D. Piorkowski, D. Reimer, J. Richards, J. Tsay, K. R. Varshney, FactSheets: Increasing trust in AI services through supplier’s declarations of conformity, IBM J. Res. Dev. 63 (4/5) (2019) 6:1–6:13. doi:10.1147/JRD.2019.294 2288.

[14] I. D. Raji, A. Smart, R. N. White, M. Mitchell, T. Gebru, B. Hutchinson, J. Smith-Loud, D. Theron, P. Barnes, Closing the AI accountability gap: defining an end-to-end framework for internal algorithmic auditing, in: Proceedings of the 2020 Conference on Fairness, Accountability, and Transparency, 2020, pp. 33–44. doi:10.1145/3351 095.3372873.

[15] T. Lebo, S. Sahoo, D. McGuinness, PROV-O: The PROV Ontology, W3C Recommendation, https://www.w3.o rg/TR/prov-o/ (2013).

[16] OpenLineage Project, OpenLineage Specification, https: //openlineage.io/docs/spec/, accessed 13 October 2026 (2026).

[17] S. Torres-Arias, H. Afzali, T. K. Kuppusamy, R. Curtmola, J. Cappos, in-toto: Providing Farm-to-Table Guarantees for Bits and Bytes, in: Proceedings of the 28th USENIX Security Symposium, 2019, pp. 1393–1410.

[18] Open Source Security Foundation, Supply-chain Levels for Software Artifacts (SLSA) Specification, Version 1.0, https://slsa.dev/spec/v1.0/, accessed 13 October 2026 (2023).

[19] Z. Newman, J. S. Meyers, S. Torres-Arias, Sigstore: Software Signing for Everybody, in: Proceedings of the 2022 ACM SIGSAC Conference on Computer and Communications Security, 2022, pp. 2353–2367. doi:10.1145/35 48606.3560596.

[20] A. Mokhov, N. Mitchell, S. Peyton Jones, Build systems à la carte, Proc. ACM Program. Lang. 2 (ICFP) (2018) 1–29. doi:10.1145/3236774.

[21] G. Rothermel, M. J. Harrold, A safe, eficient regression test selection technique, ACM Trans. Softw. Eng. Methodol. 6 (2) (1997) 173–210. doi:10.1145/248233 .248262.

[22] M. Gligoric, L. Eloussi, D. Marinov, Practical regression test selection with dynamic file dependencies, in: Proceedings of the 2015 International Symposium on Software Testing and Analysis, 2015, pp. 211–222. doi: 10.1145/2771783.2771784.

[23] C.-P. Bezemer, S. McIntosh, B. Adams, D. M. German, A. E. Hassan, An empirical study of unspecified dependencies in make-based build systems, Empir. Softw. Eng. 22 (6) (2017) 3117–3148. doi:10.1007/s10664-017 -9510-8.

[24] N. Licker, A. Rice, Detecting Incorrect Build Rules, in: 2019 IEEE/ACM 41st International Conference on Software Engineering (ICSE), 2019, pp. 1234–1244. doi: 10.1109/ICSE.2019.00125.

[25] S. Spall, N. Mitchell, S. Tobin-Hochstadt, Build scripts with perfect dependencies, Proc. ACM Program. Lang. 4 (OOPSLA) (2020) 1–28. doi:10.1145/3428237.

[26] D. Sculley, G. Holt, D. Golovin, E. Davydov, T. Phillips, D. Ebner, V. Chaudhary, M. Young, J.-F. Crespo, D. Dennison, Hidden Technical Debt in Machine Learning Systems, In: Advances in Neural Information Processing Systems 28, Curran Associates (2015). URL https://proceedings.neurips.cc/paper\_f iles/paper/2015/hash/86df7dcfd896fcaf2674f 757a2463eba-Abstract.html

[27] E. Breck, S. Cai, E. Nielsen, M. Salib, D. Sculley, The ML test score: A rubric for ML production readiness and technical debt reduction, in: 2017 IEEE International Conference on Big Data (Big Data), 2017, pp. 1123–1132. doi:10.1109/BigData.2017.8258038.

[28] N. Polyzotis, S. Roy, S. E. Whang, M. Zinkevich, Data Management Challenges in Production Machine Learning, in: Proceedings of the 2017 ACM International Conference on Management of Data, 2017, pp. 1723–1726. doi: 10.1145/3035918.3054782.

[29] S. Schelter, D. Lange, P. Schmidt, M. Celikel, F. Biessmann, A. Grafberger, Automating large-scale data quality verification, Proc. VLDB Endow. 11 (12) (2018) 1781– 1794. doi:10.14778/3229863.3229867.

[30] D. Kreuzberger, N. Kühl, S. Hirschl, Machine Learning Operations (MLOps): Overview, Definition, and Architecture, IEEE Access 11 (2023) 31866–31879. doi: 10.1109/ACCESS.2023.3262138.

[31] S. Idowu, D. Strüber, T. Berger, Asset Management in Machine Learning: State-of-research and State-of-practice, ACM Comput. Surv. 55 (7) (2023) 1–35. doi:10.1145/ 3543847.

[32] A. Barrak, E. E. Eghan, B. Adams, On the Co-evolution of ML Pipelines and Source Code - Empirical Study of DVC Projects, in: 2021 IEEE International Conference on Software Analysis, Evolution and Reengineering (SANER), 2021, pp. 422–433. doi:10.1109/SANER50967.2021. 00046.

[33] J. F. Pimentel, L. Murta, V. Braganholo, J. Freire, A Large-Scale Study About Quality and Reproducibility of Jupyter

Notebooks, in: 2019 IEEE/ACM 16th International Conference on Mining Software Repositories (MSR), 2019, pp. 507–517. doi:10.1109/MSR.2019.00077.

[34] Cook County Assessor’s Ofice, model-res-avm: Residential Automated Valuation Model, GitHub repository, https://github.com/ccao-data/model-res-avm, accessed 13 October 2026 (2026).

[35] E. Kalliamvakou, G. Gousios, K. Blincoe, L. Singer, D. M. German, D. Damian, The promises and perils of mining GitHub, in: Proceedings of the 11th Working Conference on Mining Software Repositories, 2014, pp. 92–101. doi: 10.1145/2597073.2597074.