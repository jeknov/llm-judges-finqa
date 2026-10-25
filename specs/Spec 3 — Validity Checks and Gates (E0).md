# Spec 3 — Validity Checks and Gates (E0)

Oct 22, 2026 · @Jekaterina

## Purpose

This spec decides whether the Jev scores can be used at all: it computes the comparator scores, runs the five validity analyses, and evaluates gates D1–D4, whose decisions Specs 4–6 read before running. It is Step 3 of protocol v1.

Inputs: `pool.parquet`, `verified.parquet` and `jev_scores.parquet` from Spec 1, including the `#neg` validation negatives and the `fixed_demo` items, and `surprisal.parquet` from Spec 2. Outputs: `scores/comparators.parquet`; `results/e0/` with one table per analysis; `results/gates.json`; `data/analysis_frame.parquet`, the single table Specs 5 and 6 read; and the H1 row of `results/hypotheses.csv`. Shared conventions and helpers are defined in Spec 1.

## Comparator scores

Five comparators are computed for every pool item; `judge_quality` is also computed for the `#neg` items, because A1 reports its AUROC. Self-consistency uses a 3,000-item stratified subsample to cap API cost. The three loss-based comparators call `answer_nll` from Spec 2 with different messages, so each needs one extra NLL pass and no new span logic.

| Column | Role | Definition |
| --- | --- | --- |
| `pvi` | H1 novelty check | `nll_sum` with context removed minus `nll_sum` with context, in the `direct` format. Context removal keeps the question and replaces paragraphs and table with the line "(no context provided)". |
| `icl_reducible` | H1 novelty check, RHO-LOSS approximation | `nll_mean` zero-shot minus `nll_mean` with 4 fixed demonstrations prepended as prior user and assistant turns in the `direct` format. |
| `ifd` | E1-C baseline | `nll_mean` with the instruction (question and context) divided by `nll_mean` of the answer alone, scored after an empty user turn. |
| `judge_quality` | E1-C baseline, AlpaGasus-style | Strong LLM rating from 0 to 5 of how accurate and useful the answer is, given the `judge` rendering; integer, temperature 0. |
| `sc_acc`, `sc_modal_share` | Self-consistency agreement | Strong LLM, 5 samples in the `reason` format; share correct and share agreeing with the modal answer. Subsample only. |

**Fixed demonstrations for `icl_reducible`.** The 4 `fixed_demo` items of the item's own dataset from Spec 1, passed to `build_messages` in a fixed order. The token budget guarantees they fit, so k is always 4.

**Judge prompt.** Written once in `prompts/judge_quality.txt` and frozen. It asks for an integer from 0 to 5 on one line, with 5 meaning the answer is correct and complete for the question and 0 meaning it is wrong or unsupported. Parse failures are retried once and otherwise stored as null.

**Subsample for self-consistency.** 1,500 items per dataset, stratified by `answer_type` and by R quintile, so that agreement with R can be estimated across its range. Store the subsample IDs.

## Validity analyses

Five analyses run on pool items, each writing one table to `results/e0/`. All correlations are Spearman, reported with Fisher-z 95% intervals; all AUROCs use a cluster bootstrap with 2,000 resamples.

**A1, reliability against correctness (D1, H1).** Evaluation set: the 1,000 originals that have a `#neg` copy, labeled 1, plus their negatives, labeled 0. Compute the AUROC of R, of R\_para and, for context, of `judge_quality`. Report overall and separately for `lookup` and `arith`. Bootstrap by resampling base items, so that each original and its negative stay together.

**A2, construct correlations (D2, D3, H1).** On pool originals only, compute ρ for these pairs: (I, acc), (R, acc), (R, I), (I, pvi), (I, icl\_reducible), (S, acc) and (I, S). Report each overall and within each dataset.

**A3, battery robustness.** ρ(R, R\_para) and ρ(I, I\_para), plus the intraclass correlation ICC(2,1) for each pair. Also report the share of items that change I × S cell when I\_para replaces I.

**A3b, alternative definitions of I (pre-specified robustness, added 25 October 2026 before any G or C measurement).** The primary I is battery v2's mean of P(yes | Q2) and P(yes | Q3). A1's AUROC of I, every A2 pair that involves I, A3 and A5 are repeated with two alternatives:

- **I\_v0:** the same mean under the pre-registered battery v0, from the full v0 run.
- **I\_Q3:** P(yes | Q3) alone, with I\_Q3\_para = P(yes | Q3′). Q3's wording is identical in every battery version, so I\_Q3 does not depend on the battery.

Gates D2–D4 are decided on the primary I only. For each alternative, report whether any gate would have reached a different decision. I\_Q3 is close to its ceiling and heavily tied (on the full pool, 61% of originals have I\_Q3 ≥ 0.95, and only 98 distinct values occur). So for A5 also report the share of items lying exactly at each dataset's I\_Q3 median, which are assigned to the high-I side as for the primary I.

**A4, agreement with self-consistency.** On the 3,000-item subsample: ρ(R, sc\_acc), and the AUROC of R for predicting whether the strong LLM's modal answer matches gold.

**A5, cell sizes (D4).** Count pool items in each of the four I × S cells within R ≥ 0.9, overall and per dataset. Here R means effective R after D1, with the threshold at its current value (0.9, or 0.8 after D4 fires). Medians of I and S are taken within each dataset over the same items, and are the medians stored in gates.json.

**Table format.** Each table has columns `analysis`, `quantity`, `subset`, `estimate`, `ci_low`, `ci_high`, `n`, so that gates and figures read one schema.

## Human audit

Two annotators label 150 pool items on the same three questions Jev answered, so that Jev's R and I can be compared with human judgment and not only with execution. The audit is reported in the paper but gates nothing.

**Sample.** Split pool originals at the within-dataset medians of R and I into four cells, and draw 37 or 38 items per cell, half from each dataset. Annotators never see Jev scores, cell membership or whether the item is in a validation set.

**Annotation task.** Exported with `export_sheets` from Spec 1 (Q3 check generation, Annotation sheets). Each row shows the `judge` rendering and the Q3 check, followed by three yes/no label columns in the Q1–Q3 wording from Spec 1 and an optional free-text note. Annotators complete a 10-item calibration round, on a separate sheet and discussed together, before the 150; calibration items are excluded from analysis. The adjudication pass is a third sheet holding only the disagreements, without either annotator's labels.

**Analysis.**

- Inter-annotator agreement: Cohen's κ per question.
- Agreement with Jev: AUROC of R against the human Q1 label, and of I against the mean of the human Q2 and Q3 labels, using items where the annotators agree.
- Disagreements are adjudicated by a third pass and reported separately; adjudicated labels are not used in the AUROC.

Output: `results/e0/audit.parquet` (raw labels) and an A6 table in the same schema as A1–A5.

## Gates D1–D4

Gates are evaluated on point estimates in a fixed order, and their decisions are written to `results/gates.json`, which the icl stage of Spec 4 and Specs 5–6 must read at start-up (gradient computation in Spec 4 does not depend on gates and may run earlier). A spec refuses to run if `derived.halt` is true. Confidence intervals are reported alongside, but they do not change decisions.

| Gate | Metric (source) | Triggered when | Automatic action |
| --- | --- | --- | --- |
| D1, validity | AUROC of R on `arith` items (A1) | < 0.70 | Set `derived.r_source_arith = "execution"`: effective R is 1 for verified `arith` items, so the reliability filter applies to `lookup` items only. Spec 6 then draws natural low-R items from `lookup` only. |
| D2, difficulty proxy | abs ρ(I, acc) (A2) | ≥ 0.7 | Set `derived.halt = true` with reason `difficulty_proxy`. A human decision to reframe or stop is required. |
| D3, separability | abs ρ(R, I) (A2) | ≥ 0.8 | Set `derived.halt = true` with reason `not_separable`. Revise Q2 and Q3, bump `battery_version`, rerun Spec 1 scoring and this spec. |
| D4, cell sizes | Smallest dataset × cell count (A5) | < 100 | Set `derived.r_threshold = 0.8`, recompute medians and recount. If any dataset × cell is still below 100, drop the R filter for cells and quintiles and set `derived.cells_r_filter = false`; Spec 5 then adds R as a covariate. |

D1's action has a cost worth stating in the paper: execution verifies nearly every arithmetic item, so after D1 fires, reliability carries no signal for arithmetic questions.

**Novelty flags, not gates.** The H1 thresholds abs ρ(I, pvi) < 0.8 and abs ρ(I, icl\_reducible) < 0.8 are evaluated and stored under `flags`. A failure does not stop the pipeline; it changes the paper's claim from a new construct to a cheaper estimator of an existing one.

**H1 verdict.** Recorded with `record_verdict("H1", "none", …)`. H1 is supported when all four hold: overall AUROC of R in A1 ≥ 0.70, abs ρ(I, acc) < 0.5, abs ρ(I, pvi) < 0.8, and abs ρ(I, icl\_reducible) < 0.8. Note that abs ρ(I, acc) between 0.5 and 0.7 fails H1 without triggering D2.

**`gates.json` schema.**

```json
{
  "battery_version": "v0",
  "computed_at": "<ISO timestamp>",
  "git_commit": "<hash>",
  "gates": {
    "D1": {"metric": "auroc_R_arith", "value": 0.0, "ci": [0.0, 0.0], "threshold": 0.70, "triggered": false},
    "D2": {"metric": "abs_rho_I_acc", "value": 0.0, "ci": [0.0, 0.0], "threshold": 0.70, "triggered": false},
    "D3": {"metric": "abs_rho_R_I", "value": 0.0, "ci": [0.0, 0.0], "threshold": 0.80, "triggered": false},
    "D4": {"metric": "min_dataset_cell_count", "value": 0, "threshold": 100, "triggered": false},
    "D5": null,
    "D6": null
  },
  "flags": {"abs_rho_I_pvi": 0.0, "abs_rho_I_icl_reducible": 0.0, "novelty_flag": false},
  "derived": {
    "halt": false,
    "halt_reason": null,
    "r_source_arith": "jev",
    "r_threshold": 0.9,
    "cells_r_filter": true,
    "medians": {"finqa": {"I": 0.0, "S": 0.0}, "tatqa": {"I": 0.0, "S": 0.0}},
    "halt_spec6": false,
    "halt_spec6_override": null,
    "single_proxy_claims": true
  }
}
```

Spec 3 writes D1–D4 and leaves D5 and D6 null; Spec 5 fills them and the last three `derived` fields. Every writer reads, updates and rewrites the whole file through one helper, `update_gates()`, which validates the schema before saving.

The medians in `derived` are the ones Spec 5 must use for cell assignment, so that cells cannot be recomputed differently downstream.

## Analysis frame

After the gates, the `frame` stage writes `data/analysis_frame.parquet`: one row per pool original, joining every score and applying every gate-derived value once. Specs 5 and 6 read only this table for item attributes, which removes any chance of them applying a gate differently.

| Column | Definition |
| --- | --- |
| `item_id`, `dataset`, `answer_type` | From Spec 1 |
| `R`, `I`, `V`, `indecision`, `R_para`, `I_para` | From Spec 1 |
| `R_eff` | 1 for verified `arith` items when `r_source_arith = "execution"`, else `R` |
| `V_eff` | `R_eff × I` |
| `S`, `acc` | From Spec 2 |
| `pvi`, `icl_reducible`, `ifd`, `judge_quality` | From this spec |
| `reliable` | `R_eff ≥ r_threshold` |
| `in_design` | `reliable` if `cells_r_filter` is true, else true for all items |
| `hiI`, `hiS` | I and S above the dataset's `gates.json` medians; null when `in_design` is false |
| `cell` | One of `hiI_hiS`, `hiI_loS`, `loI_hiS`, `loI_loS`; null when `in_design` is false |
| `quintile` | 1–5, quintile of I within dataset among `in_design` items |

The frame is rebuilt whenever `gates.json` or any input table changes, which DVC handles through the stage's dependencies.

## Acceptance tests and open items

Spec 3 is done when these pass and `gates.json` is committed.

- [ ] `comparators.parquet` has non-null `pvi`, `icl_reducible` and `ifd` for every pool item, and `judge_quality` for at least 99%.
- [ ] Self-consistency covers exactly the stored 3,000-item subsample.
- [ ] A unit test on a synthetic set with known separation (AUROC 0.8) recovers it within 0.02, and the cluster bootstrap keeps each original with its negative.
- [ ] Every A1–A6 table follows the shared schema and has a non-null `n`.
- [ ] `gates.json` validates against a JSON Schema checked into `conf/`, and its medians match a recomputation from the pool at the final `r_threshold`.
- [ ] Changing `r_threshold` in D4 triggers a recomputation of the medians and cell counts.
- [ ] The icl stage and Specs 5–6 each contain a start-up check that exits when `derived.halt` is true; a test with a halted `gates.json` confirms it.

* [ ] `judge_quality` exists for every `#neg` item, and A1 uses exactly the 1,000 originals paired with negatives.
* [ ] The analysis frame has one row per pool original and no derived items; its `cell` and `quintile` counts match a recomputation from `gates.json`; every row has `R_eff` in \[0, 1\].
* [ ] With a synthetic `gates.json` in which D1 fired, every verified `arith` row has `R_eff = 1` and every `lookup` row has `R_eff = R`.
* [ ] Concurrent writes are impossible: `update_gates()` takes a file lock, and a test with two writers confirms that neither update is lost.
* [ ] The H1 row in `hypotheses.csv` matches the rule applied to the A1 and A2 tables.

**Open items.**

- Choose `model.strong`, shared with Spec 1, and record its version string.
- Recruit two annotators for the audit and the Q3 validation in Spec 1; both tasks use the annotation sheets from Spec 1.
