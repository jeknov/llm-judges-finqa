# Spec 5 — Main Hypothesis Tests (E1, E2)

Oct 22, 2026 · @Jekaterina

## Purpose

This spec defines the selection conditions and runs the confirmatory tests of H2, H3, H4 and H6, then evaluates gates D5 and D6. It is Step 5 of protocol v1. It only builds condition files and analyzes; all model computation goes through the Spec 4 engines.

Inputs: `data/analysis_frame.parquet` and `results/gates.json` from Spec 3, and `influence.parquet`, the projected gradients and the C runs from Spec 4. Item attributes and every gate-derived label (`R_eff`, `V_eff`, `in_design`, `cell`, `quintile`) are read from the frame and never recomputed here. Shared conventions and helpers are defined in Spec 1.

All tests here are confirmatory and fixed in advance. Any analysis not listed is exploratory and must be labeled as such in the paper.

## Selection conditions

There are 16 proxy C conditions, each written to `conf/conditions/<name>.yaml` with its eligible IDs by the `e1` stage. Every rule is applied within each dataset separately on analysis-frame columns, and the two eligible lists are concatenated.

| Condition | Eligible items (per dataset, frame columns) | Draws |
| --- | --- | --- |
| `zero_shot` | none, k = 0 | 1 |
| `random` | all frame rows | 50 |
| `top_R` | top 20% by `R_eff` | 50 |
| `top_V` | top 20% by `V_eff` | 50 |
| `top_I_resid` | `in_design` rows, top 20% by the residual of I regressed on acc | 50 |
| `judge_quality` | top 20% by `judge_quality`, ties broken by `stable_seed(global_seed, item_id)` | 50 |
| `ifd` | rows with `ifd` < 1, then top 20% by `ifd` | 50 |
| `cell_hiI_hiS`, `cell_hiI_loS`, `cell_loI_hiS`, `cell_loI_loS` | rows whose `cell` equals the condition's cell | 50 each |
| `quint_1` … `quint_5` | rows whose `quintile` equals the condition's number | 50 each |

The `ifd` < 1 filter follows the original IFD method, which treats ratios at or above 1 as items where the instruction does not help.

**Residualization for `top_I_resid`.** Fit I = a + b·acc + c·acc² by ordinary least squares on `in_design` rows within the dataset, and rank by the residual. The quadratic term absorbs curvature, since acc takes only six values (0 to 5 correct out of 5).

**Recorded per condition.** Eligible count, mean R, I, S and acc of the eligible set, and its share of `arith` items. These go in a supplementary table, because a condition that differs in answer type composition could win for that reason alone.

## E1-G: influence regression

The primary H4 test is a single regression of G on all pool items, with no subsets. Uncertainty comes from a joint bootstrap over pool items and probe items, since G itself depends on which probes were drawn.

**Model.** On all analysis-frame rows, with I, S, `R_eff` and acc standardized to mean 0 and standard deviation 1 within the frame. R in the formula below is `R_eff`, which is the reliability covariate whether or not D4 dropped the R filter.

```latex
G_i = \beta_0 + \beta_1 I_i + \beta_2 S_i + \beta_3 (I_i \times S_i) + \beta_4 R_i + \beta_5\,\text{acc}_i + \gamma_{\text{dataset}(i)} + \varepsilon_i
```

**Simple slopes.** The slope of S at low I is β2 + β3 × q25(I), and at high I it is β2 + β3 × q75(I), where q25 and q75 are the standardized 25th and 75th percentiles of I.

**Bootstrap.** 1,000 replicates. In each: resample the 500 probe vectors with replacement and recompute ḡ and G for every item; resample pool items with replacement, stratified by dataset; refit the model. Report percentile 95% intervals for β3 and both simple slopes.

**H4 under G is supported when all three hold:** the interval for β3 lies above 0; the interval for the low-I slope lies below 0; and the interval for the high-I slope lies above 0. If β3 is positive but the low-I slope is not negative, report the result as positive moderation without a sign flip, which is weaker than H4.

**H2 under G.** The difference between the mean G of the `top_I_resid` eligible set and the mean G of the whole pool, with the same bootstrap. It is supported when the interval lies above 0 and the same difference against the `judge_quality` set also lies above 0.

**Robustness, reported but not confirmatory.** Refit with `G_dot` in place of G, with I\_para in place of I, and within each dataset separately.

**Alternative definitions of I (pre-specified robustness, added 25 October 2026 before any G or C measurement).** H2–H4 are re-estimated, not confirmatorily, with I replaced by **I\_v0** (the mean of Q2 and Q3 under the pre-registered battery v0) and by **I\_Q3** (P(yes | Q3) alone); see Spec 3, A3b. This covers the E1-G regression, the E2 slope, and the C conditions that depend on I (`top_V`, `top_I_resid`, the four cells and the five quintiles). Those conditions are rebuilt for each alternative, with their own medians and quintile cut points. Report one table with each hypothesis's verdict under the primary I, I\_v0 and I\_Q3, and describe any disagreement as a finding. I\_Q3 is heavily tied, and on the full pool many more items share the top-20% cutoff value than there are slots. So its top-20% and quintile selections use the tie-breaking rule set for the primary conditions (placeholder P8), and the share of each eligible set decided by tie-breaking is reported. Spec 6's high-I × high-S cell follows the same rule when E3 is repeated under I\_Q3.

## E1-C: demonstration conditions

Under proxy C the unit of analysis is a test item answered under one draw of one condition. Models are fit on in-distribution test items only. FinanceBench results are reported with the same models fit separately, and only if at least 50 items survived Spec 4's filter.

**Estimation.** Everything is in Python; no R is used. The formulas below are written in mixed-model notation for readability, but each is estimated as follows, in `fqsel/stats/fe_logit.py`, which Spec 6 reuses:

- *Test items* enter as fixed effects: a logistic regression with one intercept per test item, fit by maximum likelihood (scipy L-BFGS with an analytic gradient, vectorized over items). Every condition and draw uses the same test items, and each item has hundreds of rows, so the incidental-parameter bias is negligible. The `dataset` term is constant within an item and is absorbed by the item intercepts. Items whose outcome does not vary across the fitted rows have no finite intercept and are dropped; their number is reported.
- *Draws* enter through the bootstrap rather than as a random effect. Uncertainty comes from a two-way cluster bootstrap, 1,000 replicates: in each, resample test items with replacement, stratified by dataset, and, independently, resample draws with replacement within each condition; refit. A resampled item that appears twice gets two intercepts.
- Intervals are bootstrap percentile 95% intervals. The one-sided p-value for a coefficient with alternative > 0 is (1 + number of replicates ≤ 0) / (1 + number of replicates), and symmetrically for < 0. Holm corrections use these p-values.
- A fit is accepted only if the gradient norm falls below 1e-6; otherwise the stage fails.

**H2 under C.** On the conditions `random`, `top_R`, `top_V`, `top_I_resid`, `judge_quality` and `ifd`:

```
correct ~ condition + dataset + (1 | test_item) + (1 | condition:draw)
```

The reference level is `random`. The draw term is nested in condition, because draws of different conditions share nothing; in the bootstrap, this means draws are resampled separately within each condition. Five contrasts against `random` are tested, plus `top_I_resid` against `judge_quality`; Holm correction runs over these six. H2 under C is supported when `top_I_resid` beats both `random` and `judge_quality` after correction.

**H4 under C.** On the four cell conditions, with cell membership coded as two factors:

```
correct ~ hiI * hiS + dataset + (1 | test_item) + (1 | condition:draw)
```

The interaction coefficient is the C counterpart of β3. In addition, the effect of hiS is estimated within the low-I cells and within the high-I cells. H4 under C is supported when the interaction is positive and the two within-I effects of hiS have opposite signs, with the negative one at low I.

**Secondary outcomes, descriptive.** Accuracy gain over `zero_shot` and over `random` per condition and dataset, with bootstrap intervals over draws. The CBI rate per condition from draws 0–9, with bootstrap intervals over test items.

## E2: dose-response

H3 predicts that value rises with I across its range, not only between extremes. Quintiles are the frame's `quintile` column, numbered 1 (lowest I) to 5.

**Under G.** Regress G on the quintile index as a numeric predictor, with dataset as a fixed effect, using the joint bootstrap from E1-G. Supported when the slope interval lies above 0. Also report the mean G per quintile with intervals, to show whether the trend is monotone or driven by one end.

**Under C.** On the five quintile conditions:

```
correct ~ quintile + dataset + (1 | test_item) + (1 | condition:draw)
```

The quintile index is numeric. Supported when its coefficient is positive at α = 0.05.

**H3 overall** is supported when both proxies support it. Monotonicity is checked descriptively: report whether adjacent quintile means are ordered, without a separate test.

## Proxy agreement and gates D5, D6

H6 compares how the two proxies rank the 15 conditions that use demonstrations (every condition except `zero_shot`). Gates D5 and D6 are then appended to `results/gates.json` under the same schema as D1–D4.

**H6.** For each condition, the G score is the mean G of its eligible set, and the C score is its mean accuracy over all draws and in-distribution test items. Compute Spearman ρ between the two vectors of 15 scores. For the interval, bootstrap 1,000 times, resampling probe vectors for G and draws within each condition for C. H6 is supported when ρ ≥ 0.5.

**Gates.**

| Gate | Triggered when | Action |
| --- | --- | --- |
| D5, core claim | Both fail: the interval for mean G of `top_I_resid` minus pool mean G is not above 0 (no positive association of residualized I with G), and H4 under G is not supported (no sign flip). The comparison with `judge_quality` is not part of D5. | Set `derived.halt_spec6 = true`. Spec 6 runs only if a person sets `derived.halt_spec6_override` to a written reason. The paper becomes the E0 measurement study. |
| D6, proxy disagreement | ρ < 0.3 in H6 | Set `derived.single_proxy_claims = false`. Every result is reported under both proxies, and no claim may rest on one alone. |

Both gates are written with `update_gates()` from Spec 3, which validates the schema.

D5 is deliberately evaluated under G only. G is the primary proxy and its test does not depend on draws; requiring C as well would let the noisier proxy veto the main claim.

## Figures and output tables

Three figures are produced from the stored tables by `scripts/figures.py`, never from in-memory results, so they can be regenerated without rerunning analyses.

| Figure | Shows | Construction |
| --- | --- | --- |
| Fig. 1, headline | H4 under both proxies | x: S, binned into within-dataset deciles; y: mean G per bin with bootstrap intervals; two lines for items below and above the I median. C cell accuracies are overlaid as four points on a secondary y-axis, at the mean S of each cell. |
| Fig. 2 | H6 | Scatter of the 15 conditions, G score against C score, labeled, with ρ and its interval in the corner. |
| Fig. 3 | H3 | Mean G and mean C accuracy per I quintile, as two panels sharing the x-axis. |

**Output tables.** Written to `results/e1g/`, `results/e1c/`, `results/e2/` and `results/h6/`, in the Spec 3 schema plus two columns: `hypothesis` (H2, H3, H4 or H6) and `supported` (bool, or null for descriptive rows). Verdicts go to `results/hypotheses.csv` only through `record_verdict`, one row per hypothesis × proxy (G, C, and overall), each naming the rule applied and the table it came from. Overall verdicts follow the protocol: H2, H3 and H4 need support under both proxies; H6 has one verdict.

## Acceptance tests and open items

Spec 5 is done when these pass and `hypotheses.csv` is committed.

- [ ] Every condition file exists, its eligible counts match a recomputation from stored scores, and no eligible list contains a probe, test, `#neg` or Spec 6 item.
- [ ] Cell and quintile assignment uses the `gates.json` medians; a test that perturbs the pool and recomputes medians confirms the code reads them from `gates.json`.
- [ ] On a simulated dataset with a planted interaction (β3 = 0.2, sign-flipping simple slopes), the E1-G pipeline recovers the sign flip in at least 90% of 100 simulations.
- [ ] On simulated C runs with a planted condition effect, the fixed-effects logistic model with the two-way bootstrap recovers it, and Holm correction is applied over exactly six contrasts.
- [ ] On a small simulated dataset, `fe_logit` matches statsmodels `Logit` with explicit item dummies to within 1e-4 on every shared coefficient.
- [ ] Bootstrap replicates are seeded and reproducible.
- [ ] Figures regenerate from `results/` alone, and the D5 and D6 entries validate against the `gates.json` schema.

* [ ] Null simulations, 200 each: with no planted interaction, the E1-G pipeline declares H4 supported in at most 7% of runs; with no planted condition effect, the H2 C-model rejects in at most 7%.
* [ ] Condition files are regenerated from the frame alone; editing a frame label changes exactly the affected files.
* [ ] `hypotheses.csv` has exactly one row per hypothesis × proxy after a full run, and rerunning the stage does not duplicate rows.
* [ ] D5 is evaluated on the pool-mean contrast and the H4 rule only; a test with `top_I_resid` beating the pool but losing to `judge_quality` does not trigger D5.

**Open item.** The bootstrap refits each C model 1,000 times on up to about 300,000 rows. Pilot timing on the 200-item pilot confirms that this fits the CPU budget; the replicates are independent and run in parallel.
