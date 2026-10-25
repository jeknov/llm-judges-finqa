# Implication-Weighted Data Selection (Protocol v1)

Oct 23, 2026 · updated Oct 24, 2026 · @Jekaterina

## How to use this brief

This brief is the entry point for implementing the study in code. It summarizes protocol v1 and the six implementation specs in one place, but the specs remain authoritative: when this brief and a spec disagree, follow the spec and flag the discrepancy.

**Sources, in order of authority.**

1. The six specs, which are complete to coding level: [Spec 1 — Pool Construction and Jev Scoring](https://claude.ai/artifact/5JKwCAyWXR5ouyNk6oNQiz), [Spec 2 — Base-Model Surprisal](https://claude.ai/artifact/LKbFMdsaTPwhd2nF4FXERt), [Spec 3 — Validity Checks and Gates](https://claude.ai/artifact/PCkMKsEtwMuhsEj59XDXu6), [Spec 4 — Value Proxies G and C](https://claude.ai/artifact/LNHPBo3oJx3nQzSJrbyjjg), [Spec 5 — Main Hypothesis Tests](https://claude.ai/artifact/9MVRSY53JSWkTvvm4Yx68d), [Spec 6 — Reliability Gate and Causal Manipulation](https://claude.ai/artifact/1VwMWLzy6nWdijnrcbabTo).
2. This brief.
3. Protocol v1, the training-free research protocol, for scientific intent wherever the specs are silent.

**Rules for the implementing agent.**

- Never invent a threshold, prompt wording, seed, or file name. Every one is fixed in a spec, and changing one invalidates pre-registration. If a value is missing, stop and ask.
- Implement each shared helper exactly once (Shared conventions, below) and call it everywhere; duplicated prompt or scoring logic is the most likely source of silent errors.
- Treat the items under "Open items" as blocking: confirm each against the real files or documentation before writing code that depends on it.
- Build in the order under "Build order", and do not start a stage until the previous stage's acceptance tests pass on the tiny profile.

## Current status (24 October 2026)

Milestones 1–3 of the build order are done: the shared helpers, all of Spec 1 (pool, verification, Q3 checks, negatives, batched Jev scoring) and the CSV annotation workflow. 351 automated tests pass, plus checks on the downloaded data. Every spec acceptance item is tracked in `code/tests/ACCEPTANCE.md`.

At pilot scale, pool, verification and Q3 checks have run on real data, with real Qwen calls for the fallback checks; the Q3 validation CSVs are out to annotators. The Jev client is configured from TypeSafe's API reference and checked against live responses; the pilot Jev run is done (299 items, `jev-1.14.0`, batched; single scoring under the determinism tolerance of 0.017, raised from 0.01 after the pilot), and its analysis is in `notebooks/pilot_jev_analysis.ipynb`. The full Spec 1 run is done: 19,258 originals and 997 negatives scored by `jev-1.14.0` with battery **v2** (primary; Q2 asked without the stated answer), and with the pre-registered battery v0 as a sensitivity analysis (Q3 checks not yet human-validated). The data are described in `notebooks/data_analysis.ipynb`, and the battery decision and v0/v1/v2 comparison in `notebooks/battery_versions_analysis.ipynb`. **No scientific measurement exists yet:** no experiment, hypothesis verdict or gate outcome.

All three models are chosen; see Models and external services.

Details and progress records live elsewhere:
- **Progress against the protocol:** experiments, hypotheses and gates in `docs/implementation-progress.md`.
- **Task-level plan:** in `specs/IMPLEMENTATION_PLAN.md`, with the owner decisions log and the verification log.

**Next:** milestone 4 (Spec 2, surprisal and base-model accuracy), then milestone 5, which produces the first scientific results: E0, the H1 verdict and gates D1–D4.

## Research goal

The study tests whether a training example is valuable for adapting a model to financial QA because its answer is strongly forced by its context, and whether that property decides when surprising examples help rather than hurt. It runs without fine-tuning, measuring value with two proxies on a frozen 7–8B base model.

**Three item-level constructs.**

| Symbol | Meaning | Measured by |
| --- | --- | --- |
| R, reliability | Is the stated answer true and supported? | Jev, question Q1 |
| I, implication strength | Does the context force this answer, and does the answer entail a checkable consequence? | Jev, mean of Q2 and Q3 |
| S, surprisal | How unexpected is the gold answer to the base model? | Teacher-forced answer NLL, as a within-dataset percentile |

**Two value proxies.**

| Proxy | What it measures | Unit |
| --- | --- | --- |
| G, gradient influence | Cosine between an item's loss gradient and the mean gradient of a probe set, on fresh LoRA parameters of the frozen model | Item |
| C, demonstration gain | Test accuracy when 4 items from a selection condition are used as in-context demonstrations, over 50 random draws | Condition |

**Hypotheses** (full rules under Gates and verdicts).

- H1: R and I are valid, and I is not a difficulty proxy or a rename of an existing score.
- H2: high-I data beats random after controlling for difficulty.
- H3: value rises monotonically with I.
- H4, the central claim: I moderates S. Surprise lowers value at low I and raises it at high I, a sign flip.
- H5: high-I, high-S items with false labels have negative value.
- H6: the two proxies rank selection conditions alike.

As such, a result that reduces to "clean labels help" or "easy data helps" counts as a null result, which is why the design residualizes on difficulty and separates R from I.

## Pipeline overview

The study is one Python package, `fqsel`, run stage by stage with `python -m fqsel run <stage>`, each stage a DVC stage. Gradient influence depends only on the pool and probes, so it can run in parallel with grading; everything that selects items by score waits for the gates.

&#91;embedded content: pipeline · 10 stages across Specs 1–6\]

| Stage | Spec | Reads | Writes |
| --- | --- | --- | --- |
| `pool`, `verify`, `q3` | 1 | raw files | `data/pool.parquet`, `data/verified.parquet`, `data/q3_checks.parquet` |
| `jev` | 1 | pool, Q3 checks | `scores/jev_scores_long.parquet`, `scores/jev_requests.parquet`, `scores/jev_scores.parquet`, `results/jev_manifest.json` |
| `surprisal` | 2 | pool | `scores/surprisal.parquet`, `runs/surprisal/samples.parquet` |
| `comparators`, `e0` | 3 | Specs 1–2 outputs | `scores/comparators.parquet`, `results/e0/` |
| `gates`, `frame` | 3 | e0 tables | `results/gates.json`, `data/analysis_frame.parquet`, H1 verdict |
| `influence` | 4 | pool, probes | `scores/influence.parquet`, `grads/` |
| `icl` | 4 | condition files, gates | `runs/icl/<condition>/<draw>.parquet` |
| `e1`, `e2` | 5 | frame, G, C runs | `conf/conditions/`, `results/e1g/`, `results/e1c/`, `results/e2/`, `results/h6/`, D5–D6, verdicts |
| `e3` | 6 | frame, gates, engines | `data/e3_items.parquet`, `scores/*_e3.parquet`, `results/e3/`, verdicts |
| `figures` | 5–6 | `results/` only | Figs. 1–4 |

## Models and external services

Three models plus one non-model ground truth. Each external client sits behind an interface with a mock (`MockJev`, `MockStrongLLM`) that returns seeded, deterministic outputs, so the whole pipeline runs offline in tests.

| Component | Config key | Used in | Role |
| --- | --- | --- | --- |
| Jev, decision-only judge (TypeSafe API), pinned to `jev-1.14.0` | `jev.*` | Specs 1, 6 | Six yes/no (noul) questions per item (Q1–Q3 and paraphrases) in one batched request → R, I, indecision; manipulation check on variants |
| Base model: **OLMo 3 7B Instruct** (`allenai/Olmo-3-7B-Instruct`, revision `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`, Apache 2.0), not finance-tuned, 65,536-token context, frozen, bf16 on vLLM | `model.base` | Specs 2–6 | Answer NLL, sampled accuracy, loss-based comparators, gradients for G, generations for C |
| Strong LLM: **Qwen3-235B-A22B-Instruct-2507** via OpenRouter, pinned to the `deepinfra/fp8` endpoint, JSON-schema output, Apache 2.0 | `model.strong` | Specs 1, 3 | Fallback Q3 checks, judge-quality scores (0–5), self-consistency on a 3,000-item subsample |
| Small open model, about 0.5B (until chosen: a locally built, test-only toy model) | `profile=tiny` | Tests | End-to-end smoke test only |
| Gold programs and derivations | none | Specs 1, 3, 6 | Execution-based verification; ground truth for E0 and E3 |
| Two human annotators, labeling CSV files returned with labels | none | Specs 1, 3 | Q3 check validation (100 items) and the 150-item audit |

The base model is the only model whose identity affects results, and it must be the same checkpoint and revision hash in every stage.

**Why these models:**
- **OLMo 3:** its training data are public, so FinQA/TAT-QA/FinanceBench contamination can be checked directly. The contamination audit is still to do.
- **Qwen3-235B:** an open-weight model of a different family, so the H2 quality-judge baseline does not share the base model's biases. It also supports the temperatures and seeds the specs require; the Claude and OpenAI reasoning APIs no longer accept temperature.
- **Departure from the protocol:** the strong model departs from the protocol's "frontier API" wording. Fine-tuning appears nowhere except an optional four-run anchor outside these specs.

## Shared conventions

These are defined in full in Spec 1, Shared conventions; every stage depends on them, so implement them first.

**Items.** `item_id = "<dataset>:<source_id>"` with dataset ∈ {finqa, tatqa, financebench}. Derived items add a suffix: `#neg` (validation negatives, Spec 1), `#corrupt` and `#premdel` (Spec 6), with `is_derived = true` and `base_item_id` set. Splits: `pool` (train), `probe` (500 dev items), `fixed_demo` (4 dev items per dataset), `test` (500 per dataset plus FinanceBench). No item changes split.

**Token budget.** Pool, probe and fixed-demo items ≤ 2,048 tokens; test items ≤ 4,096; base context ≥ 16,384. This guarantees k = 4 demonstrations always fit, so no code shrinks k or truncates.

**Shared helpers.** Each exists once and is called everywhere.

| Function | Module | Contract |
| --- | --- | --- |
| `render_item(item, mode)` | `fqsel/prompts.py` | `qa` or `judge` text; tables as pipe-delimited rows |
| `build_messages(item, fmt, demos=[])` | `fqsel/prompts.py` | The only prompt assembler; `direct` or `reason` format; demonstrations as prior turns in `direct` format |
| `extract_answer(text, answer_type)` | `fqsel/prompts.py` | Last `Answer:` line; fallback to the last number for `arith` |
| `answers_match(pred, gold, answer_type)` | `fqsel/match.py` | Numbers within 0.5% relative or 0.005 absolute; percent and scale aware; spans by normalized exact match |
| `answer_nll(items, fmt, demos)` | `fqsel/nll.py` | Teacher-forced NLL over the answer-token span only |
| `make_wrong_answer(item, tag)`, `generate_check(item, answer=None)` | `fqsel/perturb.py`, `fqsel/checks.py` | Shared by Spec 1 negatives and Spec 6 variants |
| `stable_seed(*parts)` | `fqsel/ids.py` | SHA-256 based; the built-in `hash` is banned by a lint rule |
| `jev_ask(state, questions)` | `fqsel/clients/jev.py` | One batched Jev request per item; cached, retries honouring `retry-after`, logs request IDs and input tokens |
| `read_table`, `write_table` | `fqsel/io.py` | Pandera schema validation and metadata on every read and write; `read_table("items")` unions pool and E3 items |
| `update_gates()`, `record_verdict(...)` | `fqsel/gates.py`, `fqsel/verdicts.py` | Locked, schema-validated writes to `gates.json` and `hypotheses.csv` |

**Prompt formats.** `direct`: instruction "Give only the final answer, with units if any.", assistant turn `Answer: {gold_answer}`; used for NLL, gradients and proxy C. `reason`: "Reason briefly, then give the final answer on its own last line as: Answer: \<value>"; used only for sampled accuracy. Reasoning modes in the chat template are off.

**Determinism.** Global seed 20261022. Every output records git commit, config hash and model identifiers in Parquet metadata; all parameters live in Hydra configs under `conf/`.

## Stage-by-stage summary

Each subsection gives what a stage must produce and the parameters that are easiest to get wrong. Algorithms, schemas and edge cases are in the linked spec.

### Spec 1 — Pool construction and Jev scoring

- **Pool.** FinQA train plus TAT-QA train; exact dedup, then MinHash Jaccard ≥ 0.9 leakage removal against probe and test. Map fields to the pool schema (question, context sentences, table, program or derivation, gold answer and value, `answer_type` ∈ {lookup, arith}, `support_ids`, `qa_tokens`).
- **Verification.** Port the official FinQA executor; evaluate TAT-QA derivations with an AST whitelist, never `eval`. Unverified items are excluded; FinanceBench gets `verified = null`.
- **Q3 checks.** One consequence statement per pool item, true whenever the answer is true: templates keyed on the program's final operation (subtract → direction, divide → ratio vs 1, add → total exceeds the largest addend, and so on), with a strong-LLM fallback under a JSON schema. Validated on 100 items, accepted at ≥ 90%.
- **Negatives.** 1,000 `#neg` copies with wrong answers (×{0.5, 0.8, 1.25, 2}, sign flip, boolean flip, or span swap), with regenerated Q3 checks.
- **Jev.** Six frozen yes/no (noul) questions per pool item and negative, sent in one batched request per item: `jev_ask(state, questions)` posts the `judge` rendering once as the state, with question IDs `q1`, `q2`, `q3`, `q1p`, `q2p`, `q3p`, to `/v1/systemone` with `jev.model` pinned to `jev-1.14.0`. Billing is per input token, so batching bills the state once instead of six times. R = P(yes | Q1); I = mean of P(yes | Q2) and P(yes | Q3); V = R × I; indecision = max binary entropy. Same for the paraphrase battery. Cache keyed on SHA-256 of (model ID, state, canonical JSON of questions); `429` retried honouring `retry-after`. A 200-item check confirms batched and single-question answers agree, otherwise every question is sent alone; `jev_requests.parquet` tracks input tokens per request.

### Spec 2 — Base-model surprisal

- **NLL.** `direct` format via vLLM `prompt_logprobs`, summed over the answer-token span only; `S` is the within-dataset percentile of mean NLL. Cross-check 200 items against fp32 Transformers within 0.05 nats.
- **Accuracy.** 5 `reason`-format samples (temperature 0.7, top-p 0.95, 512 new tokens); `acc` = share matching gold. More than 5% unparsed samples blocks the stage.

### Spec 3 — Validity checks, gates and the analysis frame

- **Comparators.** `pvi` (NLL without minus with context), `icl_reducible` (zero-shot minus 4-fixed-demo NLL), `ifd` (conditioned over answer-only NLL), `judge_quality` (0–5, also on negatives), and strong-LLM self-consistency on 3,000 items (5 samples at temperature 0.7, top-p 0.8, top-k 20, 512 tokens; P5, owner decision).
- **Analyses A1–A6.** A1: AUROC of R on originals vs negatives, by answer type, with a paired cluster bootstrap. A2: Spearman correlations among I, R, S, acc and comparators. A3: battery robustness. A4: self-consistency agreement. A5: dataset × cell counts. A6: human audit.
- **Gates D1–D4** → `gates.json`. **Frame** → `analysis_frame.parquet`, holding `R_eff`, `V_eff`, `in_design`, `cell`, `quintile`. Downstream stages read these labels and never recompute them.

### Spec 4 — Value proxies

- **G.** PEFT LoRA rank 16, alpha 32, on q, k, v, o; A default init, B = 0; gradients of B only; batch size 1; Rademacher projection to 8,192 dimensions, applied in chunks. G = cosine with the mean probe gradient; also `G_dot`, `grad_norm`, per-dataset G. Save all 500 projected probe vectors for the bootstrap.
- **C.** `run_condition(name, eligible_ids | demo_plan, n_draws)`: 4 demonstrations per dataset per draw, drawn with `stable_seed(global_seed, c, j)`; greedy `direct` decoding, 64 new tokens; `zero_shot` baseline; CBI from 5 samples on draws 0–9, where consistent means a modal share ≥ 0.8.

### Spec 5 — Main hypothesis tests

- **16 conditions,** all top-20% rules applied per dataset on frame columns: `zero_shot`, `random`, `top_R`, `top_V`, `top_I_resid` (I residualized on acc and acc²), `judge_quality`, `ifd` (with ifd < 1), four I × S cells, and five I quintiles.
- **E1-G.** G \~ I + S + I×S + R\_eff + acc + dataset, with standardized predictors; simple slopes of S at the 25th and 75th percentiles of I; joint bootstrap over probes and items, 1,000 replicates.
- **E1-C and E2.** Logistic models with test-item fixed effects (`fe_logit`, Python only) and a two-way cluster bootstrap over test items and draws within condition, plus a dataset fixed effect; Holm correction over six H2 contrasts. H6: Spearman correlation of G and C condition scores. Gates D5 and D6. Figs. 1–3.

### Spec 6 — Reliability gate and causal manipulation

- **Base items.** Up to 300 per dataset from the `hiI_hiS` cell, plus naturally occurring low-R items.
- **Variants.** `#corrupt`: wrong answer kept in the gold's format and absent from the context. `#premdel`: delete the non-support sentence with the highest question overlap; never a table row, the gold answer or a program argument; re-verify.
- **Manipulation check.** Jev on all variants. Corruption must lower R more than I; premise deletion must lower I more than R (one-sided Wilcoxon, α = 0.01). A failed check records `manipulation_failed`.
- **Tests.** Paired Wilcoxon on ΔG; C conditions `e3_orig`, `e3_corrupt` and `e3_premdel` share a `demo_plan` draw by draw, plus `e3_lowR`; Holm over three contrasts per proxy. Fig. 4.

## Gates and verdicts

Gates are evaluated on point estimates and written through `update_gates()`. Gated stages read `derived` at start-up and exit when halted; an agent must never bypass a halt. Verdicts are written only through `record_verdict`, one row per hypothesis × proxy.

| Gate | When | Triggered when | Action |
| --- | --- | --- | --- |
| D1 | after E0 | AUROC of R on `arith` items < 0.70 | `r_source_arith = execution`: R\_eff = 1 for verified arithmetic items |
| D2 | after E0 | abs ρ(I, acc) ≥ 0.7 | Halt: I is a difficulty proxy; a human decides |
| D3 | after E0 | abs ρ(R, I) ≥ 0.8 | Halt: revise Q2 and Q3, bump `battery_version`, rerun |
| D4 | after E0 | Any dataset × cell < 100 items | Relax R threshold to 0.8; if still short, drop the R filter (`cells_r_filter = false`) |
| D5 | after E1-G | No positive pool-mean contrast for `top_I_resid` and no H4 sign flip, under G | `halt_spec6 = true`, unless overridden with a written reason |
| D6 | after H6 | ρ(G ranking, C ranking) < 0.3 | `single_proxy_claims = false`: report both proxies for everything |

| Key | Supported when |
| --- | --- |
| H1 | AUROC of R ≥ 0.70; abs ρ(I, acc) < 0.5; abs ρ(I, pvi) < 0.8; abs ρ(I, icl\_reducible) < 0.8 |
| H2 | G: `top_I_resid` beats both the pool mean and the `judge_quality` set. C: beats `random` and `judge_quality` after Holm |
| H3 | Positive quintile slope under both proxies |
| H4 | G: β3 > 0, S slope < 0 at low I and > 0 at high I, all intervals excluding 0. C: positive hiI × hiS interaction with opposite-signed hiS effects |
| H5 | Corrupted check passed; corrupted < original after Holm; under G also below the pool median, under C also below `random`. G-only support is recorded as `supported_G_only` |
| H6 | Spearman ρ ≥ 0.5 across the 15 demonstration conditions |
| E3\_causal | Premise-deletion check passed; premise-deleted < original after Holm |

Overall verdicts for H2–H5 and E3\_causal require support under both proxies.

## Testing requirements

Tests come in four layers, all in CI; each spec's acceptance list adds stage-specific checks. A stage is not done until its acceptance list passes.

1. **Unit tests** for every shared helper: an `answers_match` suite of 100 hand-written cases (percent forms, scale words, rounding, multi-span); the FinQA executor reproducing `exe_ans` on ≥ 99% of train items; `answer_nll` span tests on toy tokens and 20 hand-picked answers; and snapshot tests freezing `build_messages` output for 10 items per format.
2. **Schema and invariant tests** on every write: unique IDs, split disjointness, no probe, test, fixed-demo or derived item in any Spec 5 selection list, probabilities in \[0, 1\], and the token budget.
3. **Statistical tests** on simulated data. Planted effects must be recovered (for example, the H4 sign flip in ≥ 90% of 100 simulations), and null data must keep false positives ≤ 7% over 200 simulations, for E1-G, the C mixed models and the E3 paired tests.
4. **End-to-end smoke test**, `profile=tiny`: 60 pool, 20 probe and 20 test items per dataset, a model of about 0.5B parameters (currently a locally built toy model, test-only), and the mock clients. All stages run in under 20 minutes; outputs validate; a second run is byte-identical apart from timestamps.

Three checks catch the errors most likely to stay silent, and deserve priority:

- **G sign check:** one SGD step on high-G items must lower probe loss more than one on low-G items.
- **B = 0 identity:** the LoRA-wrapped model must match the unwrapped model to within 1e-4 nats before any backward pass.
- **Pairing:** the E3 variants must use exactly the base items of `e3_orig`, draw by draw.

## Build order

Each milestone ends with its tests passing on the tiny profile; full-scale runs start only after milestone 7 passes end to end on the tiny profile and on a 200-item pilot.

1. ✅ **Foundation.** Package skeleton, Hydra configs, `io.py` with pandera schemas, `ids.py`, `match.py`, `prompts.py`, the mock clients, and the CI harness with the smoke-test scaffold.
2. ✅ **Data, Spec 1 without Jev.** Loaders for FinQA, TAT-QA and FinanceBench; dedup and leakage removal; the token budget; splits; the executor and evaluator; Q3 templates and the LLM fallback; negatives.
3. ✅ **Jev scoring** (against `MockJev`; the real run waits for the TypeSafe items below). Batched client with caching and `retry-after` handling, the determinism and batching checks, aggregation into R and I.
4. **Base-model scoring, Spec 2.** `answer_nll` with the span logic, sampling, extraction, the S percentile.
5. **E0 and gates, Spec 3.** Comparators, A1–A6, gates D1–D4, the frame, the H1 verdict.
6. **Proxies, Spec 4.** `compute_influence` with its sanity checks, then `run_condition`, including `demo_plan`.
7. **Analyses, Spec 5.** Condition files, E1-G with the bootstrap, E1-C, E2, H6, D5, D6, Figs. 1–3.
8. **E3, Spec 6.** Variants, manipulation check, paired tests, Fig. 4.

On real data, `influence` can start as soon as milestone 2 has produced the pool and probes and the base model is set up, since it needs no gate; the critical path runs through Q3 generation, Jev scoring and the gates.

## Deviations and open items

The specs refine protocol v1 in documented places; the full table, with reasons, is in Spec 1 under "Deviations from protocol v1". The ones most likely to surprise an implementer:

- R is validated against 1,000 constructed wrong-answer copies, because execution verifies nearly every gold label.
- S is a within-dataset percentile, not raw NLL.
- Proxy C uses the `direct` format with greedy decoding, and k is always 4.
- Gate D4 counts per dataset × cell, against 100.
- The mixed models add a dataset fixed effect, and nest draw in condition, except in Spec 6's paired models, where draws are crossed.

**Implementation decisions.** Choices made while building milestones 1–3 are listed in the plan's owner decisions log, each with its reason. The ones most relevant here:
- Python only;
- human labeling through CSV files, never Argilla;
- one batched Jev request per item;
- case-insensitive span verification for TAT-QA;
- accounting negatives in TAT-QA derivations;
- templates for FinQA table-operation items.

**Open items, blocking: confirm before coding the dependent stage.**

- [x] FinQA, TAT-QA and FinanceBench field names against the downloaded files; whether TAT-QA test gold answers are public (Spec 1). *Confirmed; TAT-QA test gold is public.*
- [x] ~~From TypeSafe's API reference: the response fields and the authentication header.~~ *Resolved 2026-10-24* from https://docs.typesafe.ai/api/system-one and live responses: `Authorization: Bearer <key>`; answers in `answers.<qid>.noul` (P(yes), rounded to 0.01); reported model in `model`; request ID in the `x-typesafe-request-id` response header. `GET /v1/models` lists only the aliases `jev-latest` and `jev-preview`, both resolving to `jev-1.14.0`; the versioned ID `jev-1.14.0` is served and reported back when requested, and the study pins it.
- [ ] Ask TypeSafe whether the per-request token overhead (about 260 tokens) is billed as measured, and how long the versioned ID `jev-1.14.0` stays available (Spec 1).
- [x] `model.base` (Spec 2). *OLMo 3 7B Instruct, revision recorded; its chat template has no reasoning block.*
- [x] `model.strong` (Specs 1, 3). *Qwen3-235B-A22B-Instruct-2507 via OpenRouter.*
- [ ] The pysbd version, and its agreement with FinQA sentence indices on 50 items per dataset (Spec 6).
- [ ] Two annotators for the CSV annotation tasks (Specs 1, 3).
- [ ] Whether the GPU budget allows CBI sampling on 10 draws, or only 5 (Spec 4).
- [ ] Values the specs leave blank, still to decide:
  - judge-quality prompt wording (P3, milestone 5);
  - top-20% rounding and tie-breaking (P8, milestone 7).
  
  Already decided: the Q3 fallback prompt (P4) and self-consistency sampling (P5).
- [ ] The ~0.5B tiny-profile model (milestone 4).
- [ ] Before using the strong model: ask TypeSafe whether Jev is built on a Qwen model.
