# Protocol for the AAAI workshop paper

Oct 25, 2026 · @Jekaterina

## Status of this document

This protocol scopes the workshop paper to what can be completed without a GPU. It reuses the measurement work of [Protocol v1](<Protocol v1 (Training-Free).md>) and its specs, but it is a different paper. Protocol v1 asks whether implication strength decides when surprising data is useful for adaptation (H4). That needs surprisal (S) and the value proxies G and C, which need a GPU, so H2–H6 are **future work** here.

What remains is the fallback that Protocol v1 names under gate D5: *"a construct-validity study of whether a calibrated judge separates reliability from implication on verifiable financial QA."* This protocol extends that fallback in two directions that the newer results support:

- a comparison of judges by family and size;
- the link to the consistency–accuracy dissociation paper (Zhang, Gondara & Novikova 2026, arXiv:2610.21128; "the CBI paper" below).

Sources for every number below: notebooks `00`–`03` and the plan's verification log.

## Working title

*What the judge sees decides what it measures: auditing financial QA training labels with calibrated judges*

## Thesis

A judge's **information state**, meaning what it is shown, decides which construct it measures. Two findings show this.

1. **Separating reliability from implication.**
   - When a determinacy question can see the stated answer, the judge partly re-checks correctness. This answer leakage couples implication (I) to reliability (R).
   - Rewording the question does not remove the leakage; hiding the answer does.
   - The coupling that remains is genuine covariance between items.
2. **The consistency–accuracy blind spot.**
   - Inside the stratum where a strong model answers unanimously, any agreement-based signal is blind by construction.
   - A judge that assesses the **gold label** still separates reliable items from confident errors (CBI).
   - A judge that assesses the **model's own answer** mostly does not.

Together these give the paper's practical claim. **Calibrated judges are effective auditors of training labels, provided their information state is controlled. They are poor verifiers of model outputs.** That is the role a data-selection pipeline needs from them.

## Motivation

- **Judge scores for data selection mix two things.** Data selection for domain adaptation increasingly relies on judge-based "quality" scores, such as AlpaGasus, DEITA and QuRating. One score mixes "is this label true?" with "how strongly does this example force its consequences?". Protocol v1's thesis is that these play different roles: reliability acts as a gate, while implication drives an example's value. Neither role can be studied until the two can be measured apart.
- **Agreement-based filters have a blind spot.** Label-free quality filters, such as self-consistency, majority vote and cross-model agreement, certify whatever a model answers consistently. The CBI paper shows that on financial QA most errors of strong models are consistent but inaccurate, so these filters keep exactly the items they should flag.
- **Finance is a good test bed.** FinQA and TAT-QA come with executable programs and derivations, so every gold label can be verified by execution rather than by another model. Wrong labels can be constructed with known ground truth.
- **The motivating case** is Protocol v1's diesel example: a drone report of a destroyed production site is worth more than an institutional forecast, even though it is less reliable. Valuing data that way requires a reliable instrument for each component first, and this paper builds and tests that instrument.

## Research questions

| RQ | Question | Answered by |
| --- | --- | --- |
| **RQ1** | Does a calibrated judge measure the reliability of a training label validly, and how reliably? | Experiments 2 and 3 |
| **RQ2** | Can a judge separate reliability from implication (determinacy and consequence)? Where does the coupling between R and I come from, and what removes it? | Experiment 2 (batteries v0, v1, v2) |
| **RQ3** | Is answer leakage specific to Jev, or a general property of LLM judges, and does it depend on the judge's size and family? | Experiment 6 (judge comparison) |
| **RQ4** | How large is the consistency–accuracy blind spot (CBI) on verified financial QA, and how does it compare with the CBI paper? | Experiments 3 and 4 |
| **RQ5** | Can a judge see into that blind spot, and does it matter whether it judges the gold label or the model's answer? | Experiments 4, 5 and 7 |

## Experiments

Status codes: **done** (results exist, in a notebook), **running**, **ready** (code and tests done, not run), **to do**, **proposed** (not yet approved by the owner).

### 1. Data construction and verification — done

- **Pool:** FinQA and TAT-QA training sets after exact deduplication, a 2,048-token budget and execution-based verification. 19,269 originals (6,145 FinQA, 13,124 TAT-QA), 61% arithmetic; every original verifies.
- **Constructed negatives:** 1,000, built with numeric factors 0.5, 0.8, 1.25 and 2, sign flips, yes/no flips and span swaps, stratified by dataset and answer type. They are the only items with known false labels.
- **Q3 checks:** 20,255 consequence checks (12,055 from templates over the gold program, 8,200 from Qwen3-235B). **Not yet human-validated (to do, required).**
- **Serves:** every RQ, as the measurement substrate.

### 2. Jev scoring and battery revision — done

- **What was run:** Jev (`jev-1.14.0`) scored the full pool with Q1 (support), Q2 (determinacy), Q3 (consequence) and paraphrases of all three, under battery v0 (pre-registered) and v2 (Q2 asked without the stated answer). Battery v1 (Q2 conditioned on truth) was tested on the 200-item pilot and rejected against criteria fixed in advance.
- **Measurement quality:** single-call ICC ≈ 0.99; repeat difference 0.012; batching changes no answer (Wilcoxon p = 0.88); paraphrase ρ 0.92–0.96.
- **RQ1 results:** R separates originals from constructed wrong answers with AUROC 0.94. Arithmetic items are the hard case (0.90); sign flips and the ×0.8 factor are the hardest perturbations.
- **RQ2 results:**

  | Battery | Q2 AUROC on wrong answers (leakage) | ρ(Q1, Q2) | ρ(R, I) overall / FinQA |
  | --- | --- | --- | --- |
  | v0, Q2 sees the answer | 0.92 | 0.82 | 0.67 / 0.80 |
  | v1, "assume the answer is correct" (pilot) | 0.85 | 0.75 | 0.66 / 0.71 |
  | v2, Q2 without the answer | 0.50 by construction | 0.48 | 0.46 / 0.52 |

  The coupling splits into **answer leakage**, which hiding the answer removes and rewording does not, and **item-level covariance**, which remains (ρ(Q1, Q2) = 0.48 under v2). Q3 is nearly independent of R (ρ ≈ 0.2).
- **Disclosure:** v2 missed one pre-set pilot criterion (C2: ρ(Q1, Q2) = 0.58 > 0.50, but 0.48 on the full pool). It was kept as a disclosed exception, and v0 is reported throughout as the sensitivity analysis.

### 3. Self-consistency of a strong model — done

- **Design:** Qwen3-235B-A22B-Instruct-2507 (fp8, pinned) answered a 3,000-item stratified subsample five times each (temperature 0.7). 15,000 calls, $2.23; 0.46% unparsed.
- **RQ1 (A4):** ρ(R, sc_acc) = 0.30, and the AUROC of R for a correct modal answer is 0.68. Agreement is strong on FinQA (ρ 0.55, AUROC 0.84) and weak on TAT-QA (0.13), where R is compressed near 1.
- **Scoring finding:** exact match understates accuracy on free-text spans. TAT-QA lookup is 59% exact, 66% with numeric equivalence, 74% with token-F1 ≥ 0.8. Lenient scoring strengthens A4 without changing its direction. Suggested set-up: exact match plus numeric equivalence as primary, token-F1 as robustness (**open owner decision, blocking question 5**).
- **Construct check:** ρ(I, sc_acc) orders as I\_v0 > I (v2) > I\_Q3 under every scoring, consistent with v0's determinacy question carrying correctness information.

### 4. Consistency–accuracy quadrants: replication of the CBI paper — done

Uses the Experiment 3 samples. Consistency is computed as the CBI paper defines it: 1 − H/log n over the 5 answers, with t = 0.8. At n = 5 that threshold means unanimity. Spec 4's looser definition (modal share ≥ 0.8, at least 4 of 5) is reported alongside.

| Measure | CBI paper (GPT-5.5, Claude Opus 4.5; CFA, CMT, FinQA) | This study (Qwen3-235B; FinQA + TAT-QA, verified gold; primary scoring) |
| --- | --- | --- |
| CBI mass (Q4 share of observations) | 5.24% overall; 7.49% FinQA | 9.4% (unanimity) |
| CBI within incorrect | 60.2% | 37% (unanimity); about 55% (at least 4 of 5) |
| Certified-set error | recommended disclosure | 16.2% |
| Answer collapse among CBI | 77.7% | 100%, by construction at n = 5 |
| Threshold invariance | nearly invariant | not testable: the curve is flat from t = 0.7 because only unanimity exceeds 0.69 at n = 5 |
| Where CBI sits | higher for free-form FinQA | FinQA add/subtract/multiply programs (13–17%) and TAT-QA text spans (10–11%); about 5% for table operations and TAT-QA arithmetic |

Points the paper adds to the CBI literature:

- **CBI rates depend on the answer matcher.** Exact match gives 10.3% CBI, numeric equivalence 9.4%, near-verbatim matching 8.2%. Almost all of the difference is in TAT-QA spans, where a consistent but differently formatted correct answer looks consistently wrong. **The matcher must be reported with any CBI rate on free-form answers.**
- **Verified gold changes what an "error" is.** With execution-verified labels, part of the CBI mass can still be under-determined questions rather than model errors (Experiments 5 and 8).
- **Definition alignment:** the CBI-within-incorrect figure depends on the consistency definition (37% vs 55%). The paper reports the CBI paper's definition as primary and the 4-of-5 variant as sensitivity (**open owner decision**).

### 5. Jev inside the blind spot: judging the gold vs judging the model's answer — done

- **Judging the gold (R).** Within the 1,730 unanimous items (281 CBI), R separates reliable items from CBI with **AUROC 0.75** (FinQA 0.86, TAT-QA 0.63). Agreement-based signals score 0.5 there by construction. In a joint logit, R dominates (OR 0.33 per SD), and lookup items are 3.7× as likely to be CBI. The v2 parts of I separate only modestly (Q2 0.63, Q3 0.64), while v0's I separates almost as well as R (0.75), again the leakage signature.
- **Judging Qwen's answer.** The design was pre-specified (19:48 UTC, 25 Oct). Jev's Q1 on Qwen's modal answer separates correct from CBI with only **AUROC 0.62** (FinQA 0.59). Two thirds of confident errors get P(supported) ≥ 0.9.
- **Joint pattern within the 281 CBI items:**

  | Jev supports | Items | Share |
  | --- | --- | --- |
  | both the gold and Qwen's answer | 163 | 58% |
  | only Qwen's answer | 90 | 32% |
  | only the gold (a clean catch) | 14 | 5% |
  | neither | 14 | 5% |

  In the 90 items, mostly FinQA arithmetic, two independent models converge on an answer other than the verified gold. First examples look like ambiguous questions or suspect gold values.
- **Selection comparison (descriptive).** The unanimity filter keeps every CBI item. On FinQA, top-20% R keeps only 6.4% of CBI items, against the 20% a random selection would keep in expectation. On TAT-QA, R alone does not help, and top-20% V = R × I works best (10.4% of CBI kept).
- **Scoring cross-check:** Jev rates answers that exact match rejects but numeric or near-verbatim matching accepts the same as correct answers (median 0.98). This independently supports counting them as correct.

### 6. Judge comparison: four open-weight judges by family and size — done (analysis: notebook 04)

- **Purpose:** RQ3, and an answer to the objection that the leakage finding is specific to one commercial judge.
- **Judges.** All four were served by one provider (Parasail, via OpenRouter, no fallbacks). The small judges ran in bf16; the large judges in fp8, the only precision Parasail serves them in. Gemma 4 31B moved from its planned Novita bf16 endpoint because Novita rate-limited every request; Llama 3.3 70B moved so that both Llama judges ran side by side on one provider (owner decisions; see the implementation-progress document):

  | Family | Small (bf16) | Large (fp8) |
  | --- | --- | --- |
  | Llama | Llama 3.2 3B Instruct — core done | Llama 3.3 70B Instruct — core done |
  | Gemma | Gemma 3 4B — core and paraphrases done | Gemma 4 31B (reasoning off) — core and paraphrases done |

- **Results** (`notebooks/04_judge_comparison_analysis.ipynb`, pair-clustered bootstrap and Fisher-z intervals). In both families the large judge detects wrong answers better (AUROC of R: Llama 0.69 → 0.88, Gemma 0.71 → 0.89, Jev 0.94), and its determinacy question under v0 leaks almost as much as its support question detects (Q2 AUROC: Llama 0.61 → 0.86, Gemma 0.66 → 0.89, Jev 0.92). Relative to each judge's own correctness signal, Q2 under v0 reproduces 61% and 77% of it for the small judges and 94–99% for the large judges and Jev. Rewording (v1) helps little; hiding the answer (v2) gives 0.50 for every judge and lowers ρ(Q1, Q2) from 0.70–0.85 to 0.47–0.61. Small judges are near chance on arithmetic items (AUROC of R 0.52–0.55); capable judges agree on answer-free determinacy (ρ with Jev 0.67–0.79). Within each family, size is confounded with precision (and, for Gemma, with model generation).

- **Items and questions.** The 995 original/negative pairs (1,990 items) from Experiment 2's analysis A1. Each item gets Q1, Q3 and Q2 in its v0, v1 and v2 wordings, verbatim from the frozen batteries. Paraphrases (Q1′, Q2′ ×3, Q3′) are planned for the Gemma pair and can be added later from the cache with identical results. That's 8,955 requests per judge for the core questions, about 10.8M input tokens.
- **Score.** P(yes) comes from first-token log-probabilities: yes/no variants are summed and renormalised, and the yes/no mass is kept as a format check. A 200-item determinism check is run per judge.
- **Measures per judge:**
  - AUROC of R on negatives (RQ1);
  - Q2's AUROC on negatives under v0 and v1, the leakage measure;
  - ρ(Q1, Q2) under v0, v1 and v2, where the v2 value is the residual covariance;
  - ρ(R, I);
  - paraphrase agreement;
  - determinism.
- **Comparisons:** within each family (size), across families at similar size, and against Jev.
- **Confounds to disclose:**
  - the Gemma pair spans two generations (3 vs 4);
  - the large models differ in size across families (70B vs 31B);
  - every judge is a general-purpose instruct model scored by log-probabilities, while Jev is a decision-only judge;
  - P(yes) levels are not calibrated across judges, so only AUROCs and rank correlations are compared.
- **Cost (actual):** $2.76 in total: Gemma 3 4B $0.35 and Gemma 4 31B $1.29 with paraphrases; Llama 3.2 3B $0.31 and Llama 3.3 70B $0.81, core only.

### 7. Judges from other families on Qwen's answers — ready to run (code done; owner-approved 29 Oct)

The notebook's main limitation for Experiment 5 is that Jev may be built on a Qwen model (still unanswered by TypeSafe). If so, generator–judge convergence is partly shared training. Three open-weight judges repeat Experiment 5 on the same items: **Qwen3-30B-A3B** (same family as the generator, as a control; DekaLLM, precision not reported), **Gemma 4 31B** and **Llama 3.3 70B** (Parasail fp8). Each judge scores both the gold answer (Q1, Q2 v2, Q3) and Qwen's modal answer (Q1) on notebook 03's 1,730 unanimous items (281 CBI), in one run on one endpoint (`python -m fqsel judges-cbi`). Primary metrics: per judge, the AUROC of judging the model's answer and of judging the gold for reliable vs CBI, and their paired difference; secondary: the joint verdict table, overlap with Jev's 90 "sides with Qwen" items, and whether the Qwen judge endorses Qwen's errors more than the others. Pre-specified in the implementation plan before any score. Estimated cost about $2.

### 8. Human validation and audit — to do (no GPU; annotators needed)

- **Q3 check validation** (100 items, acceptance ≥ 90%): required before any claim about Q3 or I.
- **CBI hand audit** (sheets exported 26 Oct: 60 items, 30 per stratum, 20 double-annotated; `code/annotation/cbi_audit/`; originally planned at about 100 items): each item labelled *model error*, *scoring artefact*, *ambiguous or under-determined question*, or *questionable gold label*. The sample oversamples the 90 items where Jev sides with Qwen and also includes the 163 "both supported" and the 14 "clean catch" items. This is the CBI paper's recommended "Q4 mass on a small labelled sample". It decides how much of the CBI mass is model error and how much is item ambiguity or label noise. That decides the reading of Experiments 4 and 5, and it ties CBI to the determinacy construct.

### Out of scope for this paper (future work)

These need the GPU: surprisal (Spec 2), base-model accuracy and gate D2, the PVI and ICL-reducible comparators, proxies G and C, H2–H6, E3, and the cross-generator CBI convergence with OLMo 3. The paper states H4 as the pre-registered next step and reports E0 only.

## How the experiments answer the RQs

| RQ | Evidence | What would change the answer |
| --- | --- | --- |
| RQ1 | Exp. 2: AUROC 0.94 on constructed negatives, by perturbation type; Exp. 3: A4 agreement with self-consistency (FinQA strong, TAT-QA weak); Exp. 6: R's AUROC for four more judges | Q3 validation does not affect R. Natural label noise is untested, since every original verifies |
| RQ2 | Exp. 2: v0/v1/v2 decomposition of the R–I coupling; Exp. 3–5: v0's I tracks correctness signals (sc_acc, CBI) more than v2's | Q3 validation failing (< 90%) would weaken every claim about I |
| RQ3 | Exp. 6: leakage (Q2 AUROC under v0, v1) and residual covariance (v2) across families and sizes | If only Jev leaks, the claim narrows to "this judge"; if leakage persists at every size, the information-state claim generalises |
| RQ4 | Exp. 4: quadrant masses and disclosure fields vs the CBI paper, under three matchers and two consistency definitions | The hand audit may move part of the CBI mass to "item problem" |
| RQ5 | Exp. 5: AUROC 0.75 on the gold vs 0.62 on the model's answer; joint verdict table; Exp. 7: non-Qwen judges on Qwen's answers | If non-Qwen judges catch CBI well, the generator–judge convergence was shared training, not a general property |

## Analyses to present

**Main text:**

1. **Table 1, data:** pool, negatives, verification and the Q3 check sources (Exp. 1).
2. **Table 2, R validity:** AUROC on constructed negatives, overall, by answer type and by perturbation, for Jev and the four judges.
3. **Figure 1, headline figure: answer leakage vs information state.** Q2's AUROC on wrong answers under v0, v1 and v2, one line per judge (Jev, Llama 3B/70B, Gemma 4B/31B), ordered by size. v2 is 0.5 by construction, so the figure shows how far each judge's v0/v1 values sit above it.
4. **Table 3, separability:** ρ(Q1, Q2), ρ(R, I) overall and per dataset, and paraphrase agreement, by battery and judge.
5. **Table 4, CBI comparison with the CBI paper:** the comparison table from Exp. 4, with matcher and consistency-definition sensitivity.
6. **Figure 2, inside the blind spot:** the distribution of R (judging the gold) and of Jev on Qwen's answer, for reliable unanimous items vs CBI, with the AUROCs (0.75 vs 0.62) and the per-dataset split.
7. **Table 5, joint verdicts within CBI:** the 2 × 2 table of Exp. 5, plus the audit outcome by cell (Exp. 8).

**Appendix:**

- measurement reliability (determinism, batching, paraphrase), the v1 criteria and the v2 exception;
- A4 under the three scorers;
- I under v2, v0 and Q3-only (pre-specified robustness);
- the selection comparison (top-R / V / I vs the unanimity filter);
- per-judge determinism and yes/no mass;
- cost.

**Statistics:** AUROCs with pair-clustered bootstrap intervals; Spearman correlations with Fisher-z intervals; descriptive comparisons for the judge comparison (no hypothesis tests across four judges); all exploratory analyses labelled as such.

## Contributions

1. **A measurement framework for the parts of a training example's value**: reliability, implication (determinacy and consequence) and surprisal, each with an item-level measure. This paper validates the first two on about 20k execution-verified financial QA items, with constructed wrong labels.
2. **An answer-leakage finding with a controlled ablation.** Determinacy questions that see the answer re-check correctness. Rewording does not fix it; hiding the answer does. What remains is covariance between items. This is tested across five judges of two families and several sizes.
3. **A replication and extension of the consistency–accuracy dissociation on verified financial QA**, with two methodological additions: CBI rates depend on the answer matcher, and the consistency threshold collapses to unanimity at small n.
4. **Evidence that a judge's role decides its usefulness in the blind spot.** Judging the gold label recovers much of the CBI mass (AUROC 0.75; FinQA 0.86); judging the model's answer does not (0.62). About a third of confident errors are cases where generator and judge agree against the verified gold. Judges are therefore useful as **label auditors**, not as **output verifiers**.
5. **Released artefacts:** the verified pool, the constructed negatives, Q3 checks, the frozen batteries, all judge scores and the archived API responses.

## Novelty

- **Judge "quality" scores split into parts.** AlpaGasus, DEITA and QuRating produce one quality score. To our knowledge, no prior work separates truth from implication in judge-based data valuation, or tests whether the two can be separated.
- **The information state, not the wording, is what separates the constructs.** Earlier work on judge prompts varies wording. Here, what the judge is shown is the variable that matters, shown with pre-recorded criteria and a rejected rewording.
- **From generator–generator to generator–judge convergence.** The CBI paper shows that GPT and Claude share confident errors. This study shows that an independent calibrated judge tends to endorse a strong generator's confident errors too, and that the same judge separates them well when it assesses the gold label.
- **CBI on execution-verified labels.** Verified gold makes it possible to ask how much of the "consistent but inaccurate" mass is model error and how much is item ambiguity. The CBI paper cannot ask that with judge-scored or single-annotator labels.

## Risks and limitations to state

- **One closed judge at the centre.** Experiment 6 addresses this. Whether Jev is Qwen-based remains open (Experiment 7).
- **Constructed negatives only.** Every original verifies, so R's validity is shown for synthetic wrong answers. Experiment 8 is the closest check on natural errors.
- **Q3 checks are unvalidated** until Experiment 8's first part is done. I\_Q3 is near its ceiling and heavily tied.
- **Post-hoc battery choice.** Disclosed as above, with v0 reported in full.
- **Small n for consistency.** At n = 5 the CBI paper's threshold is unanimity, and answer collapse is 100% by construction.
- **Exploratory analyses.** Experiments 4–7 are not among Protocol v1's pre-registered hypotheses, and no gate reads them.
- **No claim about data value.** The paper does not claim that selecting by R or I improves adaptation; that is H2–H6, future work.
- **Provider dependence.** The judges run on shared provider pools, at fp8 for Qwen and bf16 for the four judges; all responses are archived.

## Open decisions

- [ ] Answer matcher for every accuracy (blocking question 5): exact + numeric as primary, token-F1 as robustness?
- [ ] CBI consistency definition: the CBI paper's (unanimity at n = 5) as primary, 4 of 5 as sensitivity?
- [ ] Paraphrases for the Llama pair (about $1.10)?
- [x] Experiment 7: approved and pre-specified (29 Oct); run pending
- [ ] Annotators for the Q3 validation and the CBI audit.
- [ ] Workshop venue, page limit and deadline: check the AAAI-27 workshop calls.
- [ ] Questions for TypeSafe: Jev's base model; how long `jev-1.14.0` stays available.
