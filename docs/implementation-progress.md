# Implementation progress — Protocol v1 (training-free)

Status as of 25 October 2026. Sources: [Protocol v1](<../specs/Protocol v1 (Training-Free).md>), [Brief](<../specs/Brief.md>), Specs 1–6 in [`specs/`](../specs/), and the task-level [Implementation Plan](../specs/IMPLEMENTATION_PLAN.md).

## Summary

The code base (`code/`, package `fqsel`) is built through milestone M3 of the Implementation Plan and has **run all of Spec 1 on the full real data**:
- 19,269 pool originals and 1,000 validation negatives;
- 20,255 Q3 checks;
- Jev scores from `jev-1.14.0`.

The milestone M4 code (Spec 2, surprisal) is written and tested locally; its full run needs a GPU node. Spec 3's self-consistency comparator is implemented and running on Qwen. All 395 automated tests pass, and so do the checks on the downloaded data. Both models are chosen: OLMo 3 7B Instruct as `model.base` and Qwen3-235B-A22B-Instruct-2507 via OpenRouter as `model.strong` (see below).

**The Jev battery was revised before any value measurement** (details in `notebooks/02_battery_versions_analysis.ipynb`):
- **Primary, battery v2:** Q2 (determinacy) is asked without the stated answer.
- **Sensitivity analysis, battery v0:** the pre-registered battery, kept and reported.
- **Rejected, battery v1:** tested on the pilot against criteria fixed in advance.
- **Robustness analysis, Q3-only I:** pre-specified.

**No scientific measurement has been made yet** in the sense of the protocol: no experiment has run, no hypothesis has been tested and no gate has been evaluated. The full-pool R and I values are real, but the gate quantities reported below are previews computed outside Spec 3's analysis code.

| Milestone | Content | Protocol v1 counterpart | Status |
| --- | --- | --- | --- |
| M0 | Repository, configs, CLI, DVC, CI, guard-rail tests | — | Done |
| M1 | Shared helpers: IDs and seeds, schemas, answer matching, prompts, gates file, verdicts, clients, model backends | Setup | Done; real Jev and strong-LLM clients in use (vLLM parity awaits the GPU node) |
| M2 | Pool, verification, Q3 checks, negatives, CSV annotation | E0 inputs; Jev battery construction | Done on the full real data; human validation of the Q3 checks pending |
| M3 | Jev battery scoring → R, I, V, indecision | Constructs R and I | Done on the full pool: battery v2 (primary) and v0 (sensitivity); pilot also under v1 |
| M4 | Base-model surprisal and accuracy | Construct S | Code and tests done (toy model); full run waits for a GPU node |
| M5 | Comparators, E0 analyses, gates D1–D4, analysis frame | E0, H1, D1–D4 | Self-consistency implemented and running; the rest not started |
| M6 | Proxies G and C | Value proxies | Not started |
| M7 | E1-G, E1-C, E2, H6, gates D5–D6 | E1, E2, H2–H4, H6, D5–D6 | Not started |
| M8 | Variants, manipulation check, paired tests | E3, H5 | Not started (stage exists as a placeholder) |
| M9 | Full runs on the GPU node, paper outputs | Timeline weeks 1–3 | Not started |

## Base model (`model.base`)

**Decision (23 October 2026): OLMo 3 7B Instruct** ([`allenai/Olmo-3-7B-Instruct`](https://huggingface.co/allenai/Olmo-3-7B-Instruct), Ai2, Apache-2.0), pinned to revision `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc` in `code/conf/model/base.yaml`. The choice was made before any real Jev score exists, so it cannot depend on results.

**Role in the protocol.** `model.base` is the frozen learner whose adaptation the study is about.
- It produces surprisal S and base-model accuracy (H1, H4, D2, D4), the PVI, ICL-reducible and IFD comparators (H1, the IFD baseline), and both value proxies, G and C (H2–H6, D5, D6).
- R and I come from Jev and do not depend on it.

**Why this model.**
- **Contamination can be audited.** OLMo 3 is fully open: its pre- and post-training data are public. Whether FinQA, TAT-QA or FinanceBench items appear in its training corpora can therefore be checked directly, rather than argued. Contamination is the main threat to a surprisal-based study, because it would deflate S and inflate accuracy and both proxies.
- **It meets every protocol constraint:**
  - open weights and 7B parameters;
  - instruct-tuned (the reasoning model is a separate "Think" variant);
  - no finance-specific training;
  - a 65,536-token context, well above the 16,384 required for four demonstrations plus a test item.
- **It is reproducible.** Public weights, data and a fixed commit allow full replication.
- **Apache 2.0.** The model is released under the Apache License 2.0, as declared on its [model card](https://huggingface.co/allenai/Olmo-3-7B-Instruct) (confirmed 24 October 2026 at the pinned revision `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`; the repository has no separate LICENSE file). Using it, fine-tuning it for the optional anchor, and publishing derived outputs with the paper raise no licensing issue.

**Checks already run** (`pytest -m real_data`):
- **Chat template:** at the pinned revision, it renders no reasoning block in either prompt format (Spec 2 acceptance S2-09).
- **Token budget:** with the model's tokenizer, it excludes fewer than 1% of items per split. The worst cases are FinQA train (49 of 6,251) and TAT-QA dev (12 of 1,668).
- **FinanceBench:** all 52 numeric items fit, so the out-of-distribution test keeps at least 50 items, as Spec 4 requires for hypothesis testing.

**Facts to report or watch:**
- **Default system prompt:** when no system message is given, the model's chat template inserts "You are a helpful function-calling AI assistant. You do not currently have access to any functions." The study uses the template unchanged, as Spec 2 requires.
- **Extended context:** the 65K context is extended with YaRN from the 8,192 tokens the model was trained on. The longest demonstration prompts (up to about 12K tokens) rely on that extension.
- **Llama 3.1 8B Instruct as a secondary model:** suggested as an exploratory robustness check of H4 under G only, not part of the pre-registered design.

**Still to do before M4:**
- the contamination audit against the released OLMo 3 training data;
- a zero-shot accuracy comparison of FinQA dev against test;
- checking PEFT and vLLM support on the GPU node.

**Proposed sentence for the paper:**

> We use OLMo 3 7B Instruct (revision `6e5971d`) as the frozen base model. It is a general-purpose instruct model with no finance-specific training, and because its full pre- and post-training data are public, we verified that no FinQA, TAT-QA or FinanceBench item appears in its training corpora (Appendix X). Its 64K-token context accommodates four in-context demonstrations for every test item.

The verification claim in this sentence holds only once the contamination audit has been run and reported in the appendix.

## Strong model (`model.strong`)

**Decision (24 October 2026): Qwen3-235B-A22B-Instruct-2507**, accessed via OpenRouter (`qwen/qwen3-235b-a22b-2507`). Every call is pinned to one provider endpoint: DeepInfra, FP8 weights, no fallbacks, and only endpoints that honour all request parameters. This is set in `code/conf/strong/default.yaml`.

**Role in the protocol.** `model.strong` is an auxiliary tool, not a measured model. It is used three times:
- **Q3 fallback checks** for lookup items and template failures, about 40% of the pool (Spec 1). These feed I, so they matter for every hypothesis that uses I.
- **The AlpaGasus-style quality judge** (Spec 3). It is the H2 baseline that residualised I must beat.
- **Self-consistency** on 3,000 items (Spec 3, E0 analysis A4). It is supporting evidence for H1.

**Why this model.**
- **Spec settings work as written.** The specs need explicit temperatures (greedy first attempt, retry at 0.7, temperature-0 judge, sampled self-consistency), seeds and JSON-schema output. Claude and OpenAI reasoning models no longer accept temperature, so they would have forced spec changes. Qwen via OpenRouter supports all of these.
- **Clearly stronger than `model.base`.** It is far above the 7B base model. A strong judge keeps the H2 baseline from being a strawman, and a strong reference makes A4 meaningful. Smaller sizes (14B and below) were rejected for that reason; 30B-A3B was the smallest defensible alternative and saved only about a dollar.
- **A different family from `model.base`.** OLMo 3 is the base model, so the quality judge does not share its training biases.
- **Open weights, Apache 2.0.** The model is released under the Apache License 2.0 ([LICENSE](https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507/blob/main/LICENSE); confirmed 24 October 2026 on the model card, revision `ac9c66cc9b46af7306746a9250f23d47083d689e`). Using it, and publishing its generated outputs with the paper, raises no licensing issue, and it can be self-hosted later if full replicability is needed.
- **Cost.** About $6 for the whole workload (~44,000 calls, ~38M input and ~4.7M output tokens, at $0.09 / $0.55 per million tokens). Budget about $15–25 including pilots and reruns, compared with about $62–125 for Claude Sonnet 5.5.

**Reproducibility settings and remaining caveats:**
- **Provider pinning:** every response's provider is checked against the pinned one. A mismatch stops the run. Each call's request ID, provider, model and token usage is logged to `.cache/logs/strong_calls.jsonl`.
- **No revision pin:** the weights are served in FP8 and OpenRouter does not expose a revision hash, so the provider could redeploy. The paper should name the provider, precision and run dates. All generated checks, judgements and samples are cached and should be published.
- **Seeds are not a guarantee:** a seed is sent with every request, but GPU serving is not guaranteed to be bit-identical.
- **Closed training data:** Qwen's training data is closed, so FinQA/TAT-QA contamination cannot be ruled out. This matters most for A4, where memorised answers would look consistently correct. Report the model's zero-shot accuracy on FinQA dev versus test alongside A4. A4 gates nothing.

**Self-consistency settings (P5, decided 24 October 2026).** Five samples per item, each a separate request:
- temperature 0.7, top-p 0.8, top-k 20, min-p 0;
- at most 512 new tokens;
- seed for sample k: `stable_seed(global_seed, item_id, "sc", k)`.

Three independent reasons support temperature 0.7:
- **Consistency within the study:** every other sampled measure in the specs uses it. That is base-model accuracy (Spec 2: 5 samples at 0.7) and the CBI samples (Spec 4), and the Q3 retry uses 0.7 too. The strong model's consistency (A4) and the base model's accuracy (H1, D2) are therefore measured alike.
- **The developers' recommendation:** Qwen's model card and default generation settings for this model specify `Temperature=0.7, TopP=0.8, TopK=20, MinP=0`. The reference model runs as its developers intend, and top-p, top-k and min-p follow the same recommendation.
- **The self-consistency literature:** the original self-consistency work (Wang et al., ICLR 2023) sampled at about 0.5–0.7. Verify the exact settings when citing.

The 512-token limit matches Spec 2's `reason` format. The pinned endpoint (`deepinfra/fp8`) accepts every one of these parameters, and the "required parameters" setting means none is silently dropped. If the pilot shows `sc_modal_share` is almost always 1, report that as a property of the measure rather than changing the temperature after seeing results. The settings live in `code/conf/strong/default.yaml` (`self_consistency`); the self-consistency computation itself is built at M5.

**Still open:**
- before relying on the judge and the checks: ask TypeSafe whether Jev is built on a Qwen model, which would couple Jev's I and the H2 judge.

**Proposed sentence for the paper:**

> Auxiliary LLM tasks (fallback consequence checks, the quality-judge baseline, and self-consistency) use Qwen3-235B-A22B-Instruct-2507, accessed through OpenRouter with a single pinned provider endpoint (DeepInfra, FP8) between <dates>, with JSON-schema-constrained decoding, fixed temperatures and per-request seeds. We chose an open-weight model from a different family than the base model, so that the quality-judge baseline does not share the base model's training biases, and we release every generated check, judgement and sample.

> For self-consistency we draw five samples per item from the strong model at temperature 0.7 (top-p 0.8, top-k 20, the developers' recommended settings), with fixed per-sample seeds. The temperature matches all other sampled measures in the study.

## Correspondence to Specs 1–6

| Spec | Protocol v1 step | What is implemented | What remains |
| --- | --- | --- | --- |
| [Spec 1 — Pool and Jev scoring](<../specs/Spec 1 — Pool Construction and Jev Scoring.md>) | Constructs R and I; verified correctness for E0 and E3 | All five stages' logic: loaders for FinQA, TAT-QA and FinanceBench; exclusions; exact and MinHash deduplication; stratified probe, fixed-demo and test splits; FinQA executor (port of the official one); TAT-QA evaluator; Q3 templates plus LLM fallback; 1,000 `#neg` validation negatives; six-question Jev battery, versioned (v0, v1, v2; v2 asks Q2 in a second request without the stated answer), pinned model `jev-1.14.0`, with determinism and batching checks and per-request token tracking; R, I, V, indecision. **Run on the full pool** (v2 primary, v0 sensitivity). 15 of 18 acceptance items fully automated | Human validation of Q3 checks (100 LLM checks; 50 per template); the sheets need re-exporting from the full pool |
| [Spec 2 — Surprisal](<../specs/Spec 2 — Base-Model Surprisal.md>) | Construct S; base-model accuracy | `answer_nll` (span boundary after "Answer:", 0 of 20 hand-picked answers adjusted with the OLMo 3 tokenizer), the `surprisal` stage (NLL, 5 `reason` samples, acc, modal answer, within-dataset S) and its tests; vLLM-vs-fp32 cross-check written as a GPU test. **The full run waits for a GPU node** | — |
| [Spec 3 — Validity and gates](<../specs/Spec 3 — Validity Checks and Gates (E0).md>) | E0; H1; gates D1–D4 | Infrastructure: `results/gates.json` schema with locked, validated writes; start-up halt check on gated stages; the verdict recorder for `hypotheses.csv`; CSV annotation module reused by the 150-item audit. **Self-consistency comparator implemented and running** (3,000 items × 5 samples). A3b (I under v0 and Q3-only) specified | `pvi`, `icl_reducible`, `ifd` (GPU), `judge_quality` (needs P3), analyses A1–A6, gate logic, analysis frame (M5) |
| [Spec 4 — Proxies G and C](<../specs/Spec 4 — Value Proxies G and C.md>) | Value proxies G and C | Prompt assembler and backends that C will use; LoRA and projection settings entered in config, with seeds decided (P1) | `compute_influence`, `run_condition` (M6) |
| [Spec 5 — Main tests](<../specs/Spec 5 — Main Hypothesis Tests (E1, E2).md>) | E1, E2; H2, H3, H4, H6; gates D5, D6 | Verdict recorder; gated stages `e1`/`e2` registered with the halt check | Conditions, regressions, fixed-effects logit, bootstrap, figures (M7) |
| [Spec 6 — E3](<../specs/Spec 6 — Reliability Gate and Causal Manipulation (E3).md>) | E3; H5 | `make_wrong_answer` and `generate_check(answer=…)` shared with Spec 1, as Spec 6 requires; `e3` stage registered with both halt checks (D5 override honoured) | Variants, manipulation check, paired tests (M8) |

Every acceptance item in the six specs has an ID and a status in [`code/tests/ACCEPTANCE.md`](../code/tests/ACCEPTANCE.md): 18 implemented, 14 partial, 43 pending (75 in total).

## Experiments E0–E3

| Experiment | Protocol v1 | Spec | Done so far | Status |
| --- | --- | --- | --- | --- |
| E0, measurement validity | AUROC of R vs verified correctness; ρ of I and R with base-model accuracy; battery agreement; self-consistency; 150-item audit; PVI and ICL-reducible comparators | 3 (inputs from 1, 2) | Verified correctness (execution); the 1,000 constructed negatives; real R and I on the full pool under v2 and v0 (plus Q3-only I); self-consistency running; CSV workflow for the audit | Jev inputs ready; surprisal and comparators pending; analysis not built |
| E1-G, influence regression | G ~ I + S + I×S + R + acc + dataset | 5 (G from 4) | None beyond shared helpers | Not started |
| E1-C, demonstration conditions | Random, top-R, top-V, top-I residualised, LLM-judge, IFD, four I×S cells | 5 (C from 4) | Fixed demonstrations (4 per dataset) and test sets (500 per dataset + FinanceBench) are sampled by the pool stage | Not started |
| E2, dose-response | Five I quintiles under G and C | 5 | None | Not started |
| E3, reliability gate and causal manipulation | Corrupted, premise-deleted and low-R variants of the high-I, high-S cell | 6 | Wrong-answer generator and check regeneration shared with Spec 1 | Not started |

## Hypotheses H1–H6

No hypothesis has a verdict yet. `results/hypotheses.csv` does not exist, and verdicts are written only through `record_verdict`, which is tested.

| Hypothesis | Protocol v1 support criterion | Experiment | Inputs implemented | Verdict |
| --- | --- | --- | --- | --- |
| H1, Jev scores valid and new | AUROC(R) ≥ 0.70; ρ(I, acc) < 0.5; ρ(I, PVI) < 0.8; ρ(I, ICL-reducible) < 0.8 | E0 | R, I (real, v2 / v0 / Q3-only), verification, negatives | Not tested |
| H2, high-I beats random after difficulty control | Residualised I beats random under G and C, and beats the LLM-judge baseline | E1 | — | Not tested |
| H3, value rises with I | Positive trend across I quintiles under both proxies | E2 | — | Not tested |
| H4, I moderates S (central claim) | Sign-flipping I × S interaction under G; matching cell ordering under C | E1-G, E1-C | — | Not tested |
| H5, confident false labels have negative value | Corrupted < original under G, below pool median; corrupted demonstrations below random under C | E3 | Wrong-answer generator | Not tested |
| H6, proxies agree | Spearman ρ ≥ 0.5 between condition rankings | E1, E2 | — | Not tested |

## Gates D1–D6

**None of the six gates has been evaluated.** Gate evaluation needs Spec 3's analysis code, surprisal and comparators (D2, D4), and proxy results (D5, D6). Real Jev scores now exist, so the D1 and D3 quantities can be previewed (below). The mechanism the gates rely on is in place and tested:
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

**Previews from the full Jev run** (computed in the notebooks, not by Spec 3's code; not gate results):
- **D1:** AUROC of R on the 995 original/negative pairs is 0.94 overall and 0.90 on arithmetic items, far above 0.70, and identical under v0 and v2.
- **D3:** ρ(R, I) is 0.46 under v2 (FinQA 0.52, TAT-QA 0.36) and 0.18 for Q3-only I. Under the pre-registered v0 it is 0.67 overall but 0.80 on FinQA. That coupling, traced to Q2 reading the stated answer, is why the battery was revised.
- **D4 outlook:** 77% of originals have R ≥ 0.9 (FinQA 59%, TAT-QA 86%), so the cells should be large.

**Early signals worth watching.** These are not gate results.
- **D1:** execution verifies every FinQA train item and 99.3% of TAT-QA arithmetic items. Any arithmetic reliability signal will therefore come from the constructed negatives, as Spec 1 anticipates.
- **D4:** FinQA has few lookup items (185 of 6,251 train programs end in a table operation). Small per-dataset cells are possible.
- **Out-of-distribution test:** only 52 of 150 FinanceBench items have a single-number answer with evidence. All 52 fit the token budget with the OLMo 3 tokenizer, so Spec 4's threshold of 50 is met, provided deduplication removes at most two.

## Verification on real data

| Check (Spec 1 target) | Result |
| --- | --- |
| Raw file hashes match `data/raw/MANIFEST` | 8/8 |
| FinQA executor reproduces `exe_ans` on train (≥ 99%) | 6,251/6,251 (100%) |
| TAT-QA verification on train: arithmetic (≥ 97%), spans (≥ 99%), counts | 99.33%, 99.99%, 100% |
| No pool item reaches Jaccard 0.9 with a probe or test item | Confirmed with an exact check, independent of MinHash, on the tiny real-data build |
| `pool.parquet` rebuilds byte for byte from raw files | Confirmed |
| Full pool build | 19,269 originals, 1,000 negatives; 245 exclusions; 0 leakage removals; every original verifies |
| Full Jev runs (v0, v2) | Every response reports `jev-1.14.0`; determinism 0.012 (≤ 0.017, single scoring); batching 0.012 (≤ 0.022, batched); invariants pass |
| `answer_nll` span with the OLMo 3 tokenizer | 20 of 20 hand-picked answers span-exact, 0 boundary adjustments |

## Changes relative to Protocol v1 made during implementation

None of these alters a pre-registered threshold or hypothesis rule. All are owner decisions recorded in the Implementation Plan's decisions log.

- **Human labeling:** CSV files sent to annotators and returned with labels, never Argilla. Spec 1 and the protocol text were updated to say so.
- **Base model:** OLMo 3 7B Instruct, chosen for its auditable training data (see Base model above). The protocol leaves the checkpoint open; this fills it.
- **Strong model:** Qwen3-235B-A22B-Instruct-2507 via OpenRouter with a pinned provider, where the protocol says "frontier API with structured output". An open-weight model was chosen so that the specs' temperatures and seeds work as written (see Strong model above).
- **Jev cost:** each item is scored in one batched request carrying all six questions, so the state is billed once instead of six times. A 200-item check against single-question requests guards against any coupling, and `jev.model` is pinned to `jev-1.14.0`. This is a Spec 1 change; the battery wording and aggregation are unchanged.
- **Jev battery v2:** the protocol froze the v0 wording before any value measurement. v0's Q2 coupled R and I (ρ(Q1, Q2) = 0.82; FinQA ρ(R, I) = 0.80), so, under D3's own remedy and before any G or C measurement, Q2 is asked without the stated answer (v2). v2 missed one pre-set criterion on the pilot (C2: 0.58 > 0.50; 0.48 on the full pool), a disclosed exception. v0 is the pre-registered sensitivity analysis and Q3-only I a pre-specified robustness analysis.
- **Determinism tolerance:** 0.017 instead of 0.01, set after the pilot showed a repeat difference of 0.016; to be disclosed as pilot-informed.
- **Answer span (Spec 2):** the boundary sits after "Answer:", so the answer's first token includes the leading space that BPE tokenizers attach.
- **Statistics:** Python only. The protocol's lme4 mixed models become the Spec 5 fixed-effects logit with a cluster bootstrap.
- **Values the specs left blank, now decided:**
  - multiply-template rule (P2);
  - MinHash settings (P6);
  - TAT-QA count verification (P7);
  - span-swap rule for negatives;
  - prompt rendering layout;
  - case-insensitive presence check for TAT-QA spans (strict matching reached only 97.98% against the 99% target);
  - accounting negatives in TAT-QA derivations;
  - templates applying to FinQA table-operation items;
  - self-consistency sampling (P5): temperature 0.7, top-p 0.8, top-k 20;
  - Q3 LLM-fallback prompt wording (P4): drafted by Claude, approved by the owner on 24 October 2026 and frozen (`code/prompts/q3_fallback.txt`).
- **Still open:** judge-quality prompt (P3) and top-20% rounding (P8). LoRA and projection seeds (P1) are set: LoRA 20261024, projection 10242026.

## Judge comparison (exploratory, added 25 October 2026)

To test whether the answer leakage found with Jev depends on the judge and its size, the Jev battery is also asked of four open-weight LLM judges through OpenRouter, with no GPU needed. Each judge answers Q1, Q3 and Q2 in its v0, v1 and v2 wordings on the 995 original/negative pairs (8,955 requests per judge), and P(yes) comes from the first answer token's log-probabilities. The command is `python -m fqsel judges` (`--judge <key>` to pick judges, `--smoke N` for a no-output check), configured in `code/conf/judges/default.yaml`. It is outside the pre-registered pipeline: no gate or hypothesis reads it. The wrapper prompt `code/prompts/judge_yesno.txt` (P9) was approved on 25 October 2026.

**Providers and precision as run (owner decisions, 25 October 2026).** The plan pinned every judge to a bf16 endpoint: Parasail for the two small models, Novita for the two large ones. Novita's shared upstream pool rate-limited every request for Gemma 4 31B for over an hour, so the owner moved Gemma 4 31B to Parasail. The owner also chose to run both Llama models side by side on Parasail, which moved Llama 3.3 70B there too; Novita was never tried for it. Parasail serves both large models only in fp8. The configuration file still names the planned endpoints; the switch is made per run with command-line overrides, and each run's manifest (`results/judges/<judge>_manifest.json`) records the endpoint actually used.

| Judge | Planned endpoint | Endpoint as run | Precision as run | Status (26 Oct, 03:30 UTC) |
| --- | --- | --- | --- | --- |
| Gemma 3 4B | Parasail, bf16 | Parasail | bf16 | **done**: core and paraphrases ($0.35) |
| Gemma 4 31B | Novita, bf16 | **Parasail** | **fp8** | **done**: core and paraphrases ($1.29) |
| Llama 3.2 3B | Parasail, bf16 | Parasail | bf16 | **done**: core ($0.31); paraphrases not run |
| Llama 3.3 70B | Novita, bf16 | **Parasail** | **fp8** | **done**: core ($0.81); paraphrases not run |

**Analysis:** `notebooks/04_judge_comparison_analysis.ipynb` (formal comparison with intervals; total cost of the four judges $2.76).

**Consequence for the analysis.** Within each family, the larger judge is also the lower-precision one, so a size effect cannot be separated from a precision effect. The two families share the same pattern (small in bf16, large in fp8) and the same provider, which keeps the comparison across families at matched size clean, and keeps provider out of the comparison altogether. The Gemma pair also spans two model generations (Gemma 3 and Gemma 4). Comparisons across judges use AUROCs and rank correlations only, not P(yes) levels.

**Paper-ready sentence.**

> All four judges were served by the same provider (Parasail, via OpenRouter, with no fallback routing): the two small judges (Llama 3.2 3B Instruct, Gemma 3 4B) in bf16, and the two large judges (Llama 3.3 70B Instruct, Gemma 4 31B) in fp8, the only precision in which that provider serves them (the bf16 endpoint we had planned for Gemma 4 31B was persistently rate-limited, and we kept all four judges on one provider); within each family, judge size is therefore confounded with numerical precision, and we compare judges only by rank-based measures (AUROC, Spearman correlation).

## Reproducing the experiments

Every finished experiment can be re-run from its archive without new API calls, and `tests/real_data/test_reproduction.py` checks this against the frozen snapshots (requires the DVC-tracked `code/runs/` and `code/.cache/`).

| Experiment | Command (from `code/`) | Snapshot |
| --- | --- | --- |
| Jev battery v2 (primary) / v0 | `python -m fqsel run jev profile=full` (battery via `jev.battery_version`) | `runs/full_v2_2026-10-25/`, `runs/full_v0_2026-10-24/` |
| Self-consistency (Qwen) | `python -m fqsel self-consistency profile=full` | `runs/self_consistency_2026-10-25/` |
| Jev on Qwen's answers | `python scripts/jev_on_qwen_answers.py` | `runs/jev_on_qwen_2026-10-25/` |
| Judge comparison (4 judges) | `python -m fqsel judges profile=full` (Gemma pair also with `judges.include_paraphrase=true`) | `runs/judges_2026-10-26/` |
| CBI audit sheets | `python -m fqsel annotate export cbi_audit profile=full` (refuses to overwrite entered labels) | `code/annotation/cbi_audit/` |
| CBI replication (Llama 3.3 70B, Gemma 4 31B) | `python -m fqsel judges-cbi --judge llama33_70b --judge gemma4_31b profile=full` | `runs/judges_cbi_2026-10-30/` |
| CBI replication (Qwen3-235B, the generator as judge) | `python -m fqsel judges-cbi --judge qwen3_235b profile=full` | `runs/judges_cbi_qwen_2026-10-30/` |

## Blockers and next steps

**Blockers:**
- **GPU node access:** needed for Spec 2's full run and Specs 4 and 6 (vLLM, bf16).
- **Q3 human validation:**
  - two annotators;
  - re-exported sheets from the full pool. The pilot sheet was open in a spreadsheet, so it was left untouched.
- **P3 (judge-quality prompt):** needed for `judge_quality` in milestone 5.
- **P8 (top-20% rounding and tie-breaking):** needed by milestone 7. It decides whole selections under Q3-only I, because all 720 FinQA top-20% slots tie at 0.99.
- **A DVC remote:** for the archive of results and API responses.
- **Questions for TypeSafe:**
  - whether the ~260-token request overhead is billed;
  - how long `jev-1.14.0` stays available;
  - whether Jev is built on Qwen.

**Next:**
- **When self-consistency finishes:** record it.
- **CBI replication with judges (Experiment 7):** done. Llama 3.3 70B, Gemma 4 31B and Qwen3-235B (the generator's own weights, replacing Qwen3-30B, which DekaLLM's upstream rate limit blocked on two days) ran on 30 Oct for $1.93 in total, archived and analysed in `notebooks/03b_cbi_judges_analysis.ipynb`.
- **CBI hand audit:** sheets exported (`code/annotation/cbi_audit/`, 40 rows per annotator, with `INSTRUCTIONS.md`); waiting for annotators.
- **Judge comparison:** runs done and archived (`code/runs/judges_2026-10-26/`); analysed in `notebooks/04_judge_comparison_analysis.ipynb`. Optional: Llama paraphrases (about $1.10).
- **Milestone 4's full run:** on the GPU node.
- **Then milestone 5:** the remaining comparators, E0, the H1 verdict and gates D1–D4. These are the first scientific results.
