# Implementation progress — Protocol v1 (training-free)

Status as of 23 October 2026. Sources: [Protocol v1](<../specs/Protocol v1 (Training-Free).md>), [Brief](<../specs/Brief.md>), Specs 1–6 in [`specs/`](../specs/), and the task-level [Implementation Plan](../specs/IMPLEMENTATION_PLAN.md).

## Summary

The code base (`code/`, package `fqsel`) is built up to milestone M3 of the Implementation Plan. That covers all of Spec 1: the item pool, verification, Q3 checks, validation negatives and Jev scoring. It also covers the shared infrastructure every later spec relies on. All 325 automated tests pass, and so do the checks on the downloaded data.

So far the pipeline has run only with a test-only toy model and mock Jev and strong-LLM clients. **No scientific measurement has been made yet:** no experiment has run on real scores, no hypothesis has been tested and no gate has been evaluated. The current R and I values come from `MockJev` and are placeholders.

| Milestone | Content | Protocol v1 counterpart | Status |
| --- | --- | --- | --- |
| M0 | Repository, configs, CLI, DVC, CI, guard-rail tests | — | Done |
| M1 | Shared helpers: IDs and seeds, schemas, answer matching, prompts, gates file, verdicts, clients, model backends | Setup | Done (real Jev and strong-LLM clients blocked, see Blockers) |
| M2 | Pool, verification, Q3 checks, negatives, CSV annotation | E0 inputs; Jev battery construction | Done on the tiny profile; human validation pending |
| M3 | Jev battery scoring → R, I, V, indecision | Constructs R and I | Done against `MockJev`; real run blocked |
| M4 | Base-model surprisal and accuracy | Construct S | Not started |
| M5 | Comparators, E0 analyses, gates D1–D4, analysis frame | E0, H1, D1–D4 | Not started |
| M6 | Proxies G and C | Value proxies | Not started |
| M7 | E1-G, E1-C, E2, H6, gates D5–D6 | E1, E2, H2–H4, H6, D5–D6 | Not started |
| M8 | Variants, manipulation check, paired tests | E3, H5 | Not started (stage exists as a placeholder) |
| M9 | Full runs on the GPU node, paper outputs | Timeline weeks 1–3 | Not started |

## Correspondence to Specs 1–6

| Spec | Protocol v1 step | What is implemented | What remains |
| --- | --- | --- | --- |
| [Spec 1 — Pool and Jev scoring](<../specs/Spec 1 — Pool Construction and Jev Scoring.md>) | Constructs R and I; verified correctness for E0 and E3 | All five stages' logic: loaders for FinQA, TAT-QA and FinanceBench; exclusions; exact and MinHash deduplication; stratified probe, fixed-demo and test splits; FinQA executor (port of the official one); TAT-QA evaluator; Q3 templates plus LLM fallback; 1,000 `#neg` validation negatives; six-question Jev battery with determinism check; R, I, V, indecision. 14 of 16 acceptance items fully automated | Human validation of Q3 checks (100 LLM checks; 50 per template); real Jev client (TypeSafe docs); full-size pool build (needs the `model.base` tokenizer) |
| [Spec 2 — Surprisal](<../specs/Spec 2 — Base-Model Surprisal.md>) | Construct S; base-model accuracy | Prerequisites only: `build_messages` with the `direct` and `reason` formats, `extract_answer`, `answers_match`, and the HF and vLLM backends with token log-probabilities | `answer_nll`, sampling, S percentile (M4) |
| [Spec 3 — Validity and gates](<../specs/Spec 3 — Validity Checks and Gates (E0).md>) | E0; H1; gates D1–D4 | Infrastructure: `results/gates.json` schema with locked, validated writes; start-up halt check on gated stages; the verdict recorder for `hypotheses.csv`; CSV annotation module reused by the 150-item audit | Comparators, analyses A1–A6, gate logic, analysis frame (M5) |
| [Spec 4 — Proxies G and C](<../specs/Spec 4 — Value Proxies G and C.md>) | Value proxies G and C | Prompt assembler and backends that C will use; LoRA and projection settings entered in config (seeds are placeholders) | `compute_influence`, `run_condition` (M6) |
| [Spec 5 — Main tests](<../specs/Spec 5 — Main Hypothesis Tests (E1, E2).md>) | E1, E2; H2, H3, H4, H6; gates D5, D6 | Verdict recorder; gated stages `e1`/`e2` registered with the halt check | Conditions, regressions, fixed-effects logit, bootstrap, figures (M7) |
| [Spec 6 — E3](<../specs/Spec 6 — Reliability Gate and Causal Manipulation (E3).md>) | E3; H5 | `make_wrong_answer` and `generate_check(answer=…)` shared with Spec 1, as Spec 6 requires; `e3` stage registered with both halt checks (D5 override honoured) | Variants, manipulation check, paired tests (M8) |

Every acceptance item in the six specs has an ID and a status in [`code/tests/ACCEPTANCE.md`](../code/tests/ACCEPTANCE.md): 15 implemented, 6 partial, 52 pending (73 in total).

## Experiments E0–E3

| Experiment | Protocol v1 | Spec | Done so far | Status |
| --- | --- | --- | --- | --- |
| E0, measurement validity | AUROC of R vs verified correctness; ρ of I and R with base-model accuracy; battery agreement; self-consistency; 150-item audit; PVI and ICL-reducible comparators | 3 (inputs from 1, 2) | Verified correctness (execution); the 1,000 constructed negatives that make the AUROC computable; both batteries scored (mock); CSV workflow for the audit | Inputs partly ready; analysis not built |
| E1-G, influence regression | G ~ I + S + I×S + R + acc + dataset | 5 (G from 4) | None beyond shared helpers | Not started |
| E1-C, demonstration conditions | Random, top-R, top-V, top-I residualised, LLM-judge, IFD, four I×S cells | 5 (C from 4) | Fixed demonstrations (4 per dataset) and test sets (500 per dataset + FinanceBench) are sampled by the pool stage | Not started |
| E2, dose-response | Five I quintiles under G and C | 5 | None | Not started |
| E3, reliability gate and causal manipulation | Corrupted, premise-deleted and low-R variants of the high-I, high-S cell | 6 | Wrong-answer generator and check regeneration shared with Spec 1 | Not started |

## Hypotheses H1–H6

No hypothesis has a verdict yet. `results/hypotheses.csv` does not exist, and verdicts are written only through `record_verdict`, which is tested.

| Hypothesis | Protocol v1 support criterion | Experiment | Inputs implemented | Verdict |
| --- | --- | --- | --- | --- |
| H1, Jev scores valid and new | AUROC(R) ≥ 0.70; ρ(I, acc) < 0.5; ρ(I, PVI) < 0.8; ρ(I, ICL-reducible) < 0.8 | E0 | R, I (mock), verification, negatives | Not tested |
| H2, high-I beats random after difficulty control | Residualised I beats random under G and C, and beats the LLM-judge baseline | E1 | — | Not tested |
| H3, value rises with I | Positive trend across I quintiles under both proxies | E2 | — | Not tested |
| H4, I moderates S (central claim) | Sign-flipping I × S interaction under G; matching cell ordering under C | E1-G, E1-C | — | Not tested |
| H5, confident false labels have negative value | Corrupted < original under G, below pool median; corrupted demonstrations below random under C | E3 | Wrong-answer generator | Not tested |
| H6, proxies agree | Spearman ρ ≥ 0.5 between condition rankings | E1, E2 | — | Not tested |

## Gates D1–D6

**None of the six gates has been evaluated.** Gate evaluation needs real Jev scores (D1–D4), surprisal and comparators (D2, D4), and proxy results (D5, D6), none of which exist yet. The mechanism the gates rely on is in place and tested:
- **The gates file:** `results/gates.json` follows the Spec 3 schema, and every write is locked and schema-validated.
- **Halt checks:** every gated stage (`icl`, `e1`, `e2`, `e3`) exits at start-up when `derived.halt` is true. `e3` also exits when D5 has fired and no override reason is set.

| Gate | Protocol v1 condition and action | Spec version (what the code will apply) | When | Outcome |
| --- | --- | --- | --- | --- |
| D1, validity | AUROC(R) < 0.70 on arithmetic items → restrict Q1 to lookup items or use program execution | Same threshold, measured on originals vs `#neg` copies; R is set to 1 for verified arithmetic items (`r_source_arith = execution`) | After E0 (M5) | Not evaluated |
| D2, difficulty proxy | ρ(I, acc) ≥ 0.7 → reframe or kill | abs ρ ≥ 0.7 → `halt`, human decision | After E0 (M5) | Not evaluated |
| D3, separability | ρ(R, I) ≥ 0.8 → revise Q2 and Q3 before E1 | abs ρ ≥ 0.8 → `halt`, revise battery, bump `battery_version`, rescore | After E0 (M5) | Not evaluated |
| D4, cell sizes | Any I × S cell with R ≥ 0.9 below 200 items → relax to R ≥ 0.8 | Any dataset × cell below 100 → relax to 0.8; if still short, drop the R filter | After E0 (M5) | Not evaluated |
| D5, core claim | No positive residualised-I association and no sign flip → report E0 as a measurement paper | Same, under G only; sets `halt_spec6` unless overridden with a reason | After E1-G (M7) | Not evaluated |
| D6, proxy agreement | Ranking ρ < 0.3 → report both proxies; no single-proxy claim | Same; sets `single_proxy_claims = false` | After H6 (M7) | Not evaluated |

**Early signals worth watching.** These are not gate results.
- **D1:** execution verifies every FinQA train item and 99.3% of TAT-QA arithmetic items. Any arithmetic reliability signal will therefore come from the constructed negatives, as Spec 1 anticipates.
- **D4:** FinQA has few lookup items (185 of 6,251 train programs end in a table operation). Small per-dataset cells are possible.
- **Out-of-distribution test:** only 52 of 150 FinanceBench items have a single-number answer with evidence, before the token budget. Spec 4 needs ≥ 50 survivors to hypothesis-test out-of-distribution results.

## Verification on real data

| Check (Spec 1 target) | Result |
| --- | --- |
| Raw file hashes match `data/raw/MANIFEST` | 8/8 |
| FinQA executor reproduces `exe_ans` on train (≥ 99%) | 6,251/6,251 (100%) |
| TAT-QA verification on train: arithmetic (≥ 97%), spans (≥ 99%), counts | 99.33%, 99.99%, 100% |
| No pool item reaches Jaccard 0.9 with a probe or test item | Confirmed with an exact check, independent of MinHash, on the tiny real-data build |
| `pool.parquet` rebuilds byte for byte from raw files | Confirmed |

## Changes relative to Protocol v1 made during implementation

None of these alters a pre-registered threshold or hypothesis rule. All are owner decisions recorded in the Implementation Plan's decisions log.

- **Human labeling:** CSV files sent to annotators and returned with labels, never Argilla. Spec 1 and the protocol text were updated to say so.
- **Statistics:** Python only. The protocol's lme4 mixed models become the Spec 5 fixed-effects logit with a cluster bootstrap.
- **Values the specs left blank, now decided:**
  - multiply-template rule (P2);
  - MinHash settings (P6);
  - TAT-QA count verification (P7);
  - span-swap rule for negatives;
  - prompt rendering layout;
  - case-insensitive presence check for TAT-QA spans (strict matching reached only 97.98% against the 99% target);
  - accounting negatives in TAT-QA derivations;
  - templates applying to FinQA table-operation items.
- **Still open:**
  - LoRA and projection seeds (P1), judge-quality prompt (P3), self-consistency temperature (P5) and top-20% rounding (P8);
  - the drafted Q3 fallback prompt (P4) awaits approval.

## Blockers and next steps

**Blockers:**
- Choice of `model.base` (needed for a full-size pool build and for M4).
- Choice of `model.strong`.
- TypeSafe Jev API documentation.
- Approval of the Q3 fallback prompt.
- Two annotators for the Q3 validation rounds.

**Next:** M4 (Spec 2, surprisal S and base-model accuracy), then M5, which produces the first scientific results: E0, the H1 verdict and gates D1–D4.
