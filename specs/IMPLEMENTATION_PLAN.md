# Implementation Plan — `fqsel`

Source of truth: `specs/Spec 1–6` > `specs/Protocol v1.md` (implementation brief) > `specs/Implication-Weighted Data Selection … Protocol v1 (Training-Free).md`.
Rule: a milestone starts only when the previous one passes its tests on `profile=tiny`.
Full-scale runs start only after M7 passes end to end on tiny + a 200-item pilot.

---

## Testing policy

Every major step ships with tests that confirm it is **correct** (does what the spec says) and **accurate** (its numbers agree with an independent reference). A step is not done until its tests pass and the documentation below is updated.

### Rules
1. **Tests are written with the code, not after.** A module's PR/commit contains its tests. No stage runs on real data until its tests pass on `profile=tiny`.
2. **Every spec acceptance item becomes a test.** Each item in the specs' "Acceptance tests" lists gets an ID (`S1-01` … `S6-12`) in `code/tests/ACCEPTANCE.md`, and the test that covers it carries `@pytest.mark.spec("S1-01")`. A meta-test fails if any ID has no test or a test points to an unknown ID.
3. **Tests beyond the specs are marked "(added)" below.** They check implementation correctness only: exact equalities, invariants, agreement with a reference library, or the tolerances the specs already give. They never introduce a new scientific threshold. Where a pure numerical tolerance is needed (e.g. float comparison against scipy), it is labeled a *test tolerance* in the test, not an analysis parameter.
4. **Independent references.** Where possible, a computation is checked against something not written by us: scipy/statsmodels/sklearn for statistics, the official FinQA `exe_ans` for the executor, HF Transformers fp32 for vLLM log-probs, hand-computed values for small fixtures.
5. **Four kinds of accuracy evidence**, used per step as fits:
   - *Golden fixtures*: small hand-verified inputs with expected outputs, in `code/tests/fixtures/`.
   - *Invariants on every write*: pandera schemas + checks in `write_table`, so violations fail the run, not just the test suite.
   - *Planted-effect / null simulations* for every statistical analysis.
   - *Real-data verification runs* (executor coverage, NLL cross-check, determinism check…), whose results are recorded in the Verification log below.
6. **Test markers.** `unit`, `schema`, `stats` (simulations), `e2e`, plus `gpu`, `api`, `real_data`, `slow`. CI runs everything not marked `gpu`/`api`/`real_data`. Those three run on the GPU node/with credentials before the pilot and before the full run, and their reports are saved to `results/test_reports/`.
7. **`python -m fqsel check`** (added): re-validates every existing output table against its schema and cross-table invariants (ID uniqueness, split disjointness, selection-list hygiene, gates.json schema). Run after every stage on real data.

### Owner decisions (2026-10-23) — override the brief where they differ
1. **Python only.** No R, rpy2 or lme4 anywhere; all models use `fqsel/stats/` (Spec 5 `fe_logit` + bootstrap).
2. **Human labeling uses CSV files, never Argilla** (and not XLSX). `annotate.py` writes one CSV per annotator; annotators fill in label columns and return the same CSV; `import_sheets` validates and loads it.
3. **Backends:** Hugging Face Transformers for the tiny profile and local tests; vLLM on the GPU node for pilot and full runs.
4. **E3 (`e3` stage)** is registered as a placeholder from M0 (no-op stub in the CLI and `dvc.yaml`) and implemented in M8.
5. **Values the specs leave blank** stay as explicit placeholders (see below); the owner is asked for each one when the milestone that needs it is reached.

### Placeholder policy
- Config values the specs don't fix are set to Hydra's `???` (MISSING) in `conf/`, with a comment `# OWNER DECISION PENDING — see Blocking questions #n`. Code that reads one fails loudly.
- Prompt files whose wording isn't in the specs contain only `PLACEHOLDER — awaiting owner wording`; the prompt loader refuses placeholder text outside `profile=tiny`.
- `profile=tiny` may set clearly marked test-only stand-ins (`# TEST-ONLY, not pre-registered`) so the smoke test runs with mocks.
- Test `test_no_placeholders_for_real_runs`: loading `profile=pilot` or `profile=full` fails if any `???` or placeholder prompt remains.
- When a milestone reaches a placeholder, implementation pauses and the owner is asked; the answer is recorded in the **Owner decisions log** at the bottom.

### Documentation duties (whenever a step is implemented)
- Tick the milestone checkbox here.
- Add/update the test IDs in `code/tests/ACCEPTANCE.md` (spec item → test function → status).
- Record any real-data verification result (with date, commit, number) in the **Verification log** at the bottom of this file.
- If a test reveals a spec ambiguity, add it to **Blocking questions** instead of choosing a value silently.

---

## M0 — Repository structure (first task)

**Status (2026-10-23): done.** Code lives in `code/`; `specs/` stays read-only beside it. Python 3.11 venv in `code/.venv` (created with `uv`; the system Python is 3.9). Git initialised at the project root, nothing committed yet; DVC initialised in `code/` (`dvc init --subdir`), raw data tracked with `dvc add data/raw`.

### 0.1 Version control and tooling
- [x] `git init` at the project root (DVC needs git; every output stamps the git commit)
- [x] `code/pyproject.toml`: Python 3.11, package `fqsel`, extras `[gpu]` (vllm, torch, peft, transformers) vs core (pandas, pyarrow, pandera, hydra-core, scipy, numpy, statsmodels, scikit-learn [test reference], datasketch, pysbd==<pinned>, filelock, jsonschema, matplotlib, pytest, hypothesis)
- [x] `ruff` + `pytest` config
- [ ] pre-commit hooks (deferred; ruff and pytest run in CI)
- [x] `dvc init`; `.gitignore` / `.dvcignore` for `data/`, `scores/`, `runs/`, `grads/`, `results/` (tracked by DVC, not git — except `results/gates.json` and `results/hypotheses.csv`, which are committed)
- [x] CI workflow: lint, unit, schema, stats, tiny smoke test

### 0.2 Directory layout (as built; modules for M4+ are added with their milestones)
```
jev-workshop-paper/
├── specs/                      # authoritative specs + this plan (read-only except the plan)
├── .github/workflows/ci.yml    # lint + all tests not marked gpu/api/real_data
└── code/
    ├── pyproject.toml  dvc.yaml  params.yaml  README.md
    ├── conf/                   # Hydra; no constants in code
    │   ├── config.yaml         # global_seed: 20261022, paths, defaults list
    │   ├── profile/{tiny,pilot,full}.yaml
    │   ├── model/{base,toy}.yaml          # model.base; toy = TEST-ONLY stand-in for the tiny model
    │   ├── strong/default.yaml            # model.strong
    │   ├── data/default.yaml  jev/default.yaml  annotation/default.yaml
    │   ├── proxy/influence.yaml  stats/default.yaml   # M6/M7 placeholders already entered
    │   ├── gates.schema.json   # JSON Schema for results/gates.json
    │   └── conditions/         # generated by e1 / e3 stages (M7, M8)
    ├── prompts/                # first line "# STATUS: FINAL | DRAFT | PLACEHOLDER"
    │   ├── jev_battery.yaml    # FINAL: Q1–Q3, Q1'–Q3' verbatim from Spec 1
    │   ├── q3_fallback.txt     # FINAL (P4): approved by the owner 2026-10-24
    │   └── judge_quality.txt   # PLACEHOLDER (P3)
    ├── fqsel/
    │   ├── __main__.py  cli.py # python -m fqsel run <stage> | check | annotate export|import <task>
    │   ├── config.py           # Hydra loading, config hash, placeholder policy
    │   ├── ids.py  provenance.py
    │   ├── io.py  schemas.py   # read_table / write_table, one TableDef per output file
    │   ├── prompts.py  match.py
    │   ├── ops.py              # final operation of a program/derivation (shared by checks and perturb)
    │   ├── perturb.py  checks.py  annotate.py  q3_validation.py
    │   ├── gates.py  verdicts.py  invariants.py
    │   ├── clients/            # cache.py, retry.py, jev.py (MockJev, RealJev), strong_llm.py (MockStrongLLM)
    │   ├── backends/           # base.py, hf_backend.py, vllm_backend.py, toy.py
    │   ├── data/               # common.py, finqa.py, tatqa.py, financebench.py, dedup.py, splits.py
    │   ├── verify/             # __init__ (verify_item), finqa_executor.py (ported), tatqa_eval.py
    │   └── stages/             # registry; pool, verify, q3, jev implemented; others placeholders
    ├── tests/
    │   ├── ACCEPTANCE.md       # spec acceptance item → test traceability matrix
    │   ├── unit/  schema/  e2e/  real_data/  stats/ (from M5)
    │   ├── snapshots/build_messages.json
    │   └── fixtures/           # golden/ (hand-written release-format records), synth_raw.py
    ├── data/raw/ + MANIFEST    # tracked with `dvc add data/raw`
    └── data/ scores/ runs/ grads/ results/ annotation/   # pipeline outputs
```

### 0.3 Skeleton deliverables
- [x] Empty modules with signatures + docstrings citing the spec section
- [x] `python -m fqsel run <stage>` dispatcher with a no-op for each stage, incl. the `e3` placeholder; `python -m fqsel check`
- [x] All blank-in-spec values entered as `???` placeholders in `conf/` and `prompts/` (Placeholder policy)
- [x] `dvc.yaml` with all stages, deps/outs per the Spec 1 file-layout table
- [x] `tests/ACCEPTANCE.md` with every spec acceptance item numbered

### M0 tests
- [x] `test_no_builtin_hash`: AST scan; no call to built-in `hash` in `fqsel/` (S1 acceptance)
- [x] `test_no_eval`: AST scan; no `eval`/`exec` anywhere (added; Spec 1 bans `eval` for TAT-QA)
- [x] `test_python_only` (added): no `rpy2` import and no `.R` files in the repo
- [x] `test_stage_registry` (added): CLI stages == `dvc.yaml` stages == the spec stage list incl. the `e3` placeholder
- [x] `test_no_placeholders_for_real_runs` (added): pilot/full profiles refuse to load with any `???` or placeholder prompt
- [x] `test_halt_guard`: every gated stage (`icl`, `e1`, `e2`, `e3`) exits on a halted `gates.json` (S3 acceptance)
- [x] `test_acceptance_traceability` (added): every ID in ACCEPTANCE.md has ≥ 1 test and vice versa
- [x] `test_config_strict` (added): Hydra struct mode; any config key read by code must exist in `conf/`

---

## M1 — Foundation (shared helpers)
**Status (2026-10-23): done**, except the vLLM parity test (GPU node) and the real Jev/strong-LLM clients (blocked on TypeSafe docs and the `model.strong` choice; both raise `NotImplementedError` behind the shared interface). The tiny profile uses a locally built toy model (`fqsel/backends/toy.py`, TEST-ONLY) until the owner picks the ~0.5B model at M4.
- [x] `ids.stable_seed`
- [x] `schemas.py` (one pandera model per output file) + `io.read_table/write_table` with metadata; `read_table("items")` = pool ∪ e3_items
- [x] `match.answers_match`
- [x] `prompts.render_item`, `build_messages`, `extract_answer`
- [x] `gates.update_gates`, `verdicts.record_verdict`
- [x] `clients/`: Jev + strong-LLM interfaces, `MockJev`, `MockStrongLLM`, SHA-256 cache, backoff
- [x] `backends/`: one generation/logprob interface; HF Transformers implementation (tiny profile, local Mac tests) and vLLM implementation (GPU node, pilot and full); selected by profile
- [x] Tiny smoke-test scaffold

### M1 tests
**ids**
- [x] Same value in two separate processes (S1)
- [x] (added) Matches a hand-computed SHA-256 for a fixed input; different part order → different seed

**io / schemas**
- [x] (added) Write → read round-trip preserves values, dtypes and metadata (git commit, config hash, model ids)
- [x] (added) Each schema rejects a crafted bad frame: duplicate `item_id`, probability outside [0, 1], unknown split, missing column
- [x] (added) `read_table("items")` returns pool ∪ e3 rows with no duplicate IDs

**answers_match**
- [x] 100 hand-written cases: percent forms, scale words, rounding, booleans, multi-span (S1)
- [x] (added) Boundary cases at exactly 0.5% relative and 0.005 absolute error, on both sides
- [x] (added) Property test (hypothesis): `answers_match(gold, gold)` is always true for generated numbers/spans

**prompts**
- [x] Snapshot tests: `build_messages` for 10 items × {direct, reason} (S1)
- [x] (added) Demonstrations appear as prior user/assistant turns in `direct` format, in the given order
- [x] (added) Single-assembler test: no other module constructs chat-role dicts (AST scan for `"role":`)
- [x] (added) `extract_answer` fixtures: last `Answer:` line wins; arith fallback to last number in the final 200 chars; lookup without `Answer:` → unparsed; trailing punctuation stripped

**gates / verdicts**
- [x] Two concurrent writers to `update_gates()`: neither update lost (S3)
- [x] (added) An invalid gates.json is refused and the old file is left intact
- [x] `record_verdict` upsert: rerunning does not duplicate rows (S5)

**clients**
- [x] `MockJev` and the real client on a fake HTTP endpoint pass the same interface test (S1-16)
- [x] The real Jev client passes the same interface test, with the fake endpoint serving the documented response shape and a recorded live response (2026-10-24)
- [x] (added) Cache: second identical call makes no network request (counted via a fake transport); changing model version changes the key

**backends**
- [x] (added) HF log-probs equal a manual log-softmax of the model's logits (test tolerance 1e-5)
- [ ] (added, `gpu`) HF and vLLM return the same token log-probs on the tiny model within the Spec 2 tolerance (0.05 nats)
- [x] (added) Seeded generation reproduces identical text in a fresh process

**e2e scaffold**
- [x] `profile=tiny` runs all implemented stages twice; outputs byte-identical apart from timestamps (S1)

---

## M2 — Data (Spec 1 without Jev): `pool`, `verify`, `q3`
**Status (2026-10-23): done on the tiny profile, on synthetic and on real raw data.** Outstanding: the two human annotation rounds (sheets can be exported now) and a full-size real-data pool build, which needs the `model.base` tokenizer for `qa_tokens`. Field names were confirmed against the downloaded files, and TAT-QA test gold answers are public (`tatqa_dataset_test_gold.json`).
- [x] Download raw data manually → `data/raw/`, write MANIFEST; **confirm field names** (blocking)
- [x] Loaders → pool schema; `answer_type`, `support_ids`, `qa_tokens`
- [x] Exact dedup; MinHash leakage removal (Jaccard ≥ 0.9) vs probe/test; `dedup_log.csv`
- [x] Exclusions **before** sampling; per-reason counts
- [x] Split sampling: probe 250/dataset, fixed_demo 4/dataset, test 500/dataset, FinanceBench filter
- [x] FinQA executor (ported); TAT-QA AST evaluator → `verified.parquet`
- [x] Q3 templates + strong-LLM fallback → `q3_checks.parquet`
- [x] `make_wrong_answer` → 1,000 `#neg` items with regenerated checks
- [x] `export_sheets` / `import_sheets` (CSV) for the 100-item Q3 validation

### M2 tests
**Raw data and loaders**
- [x] MANIFEST hashes match the downloaded files (S1)
- [x] (added) Golden fixtures: 4 FinQA records, 1 TAT-QA record with 6 questions (arithmetic, percent, count, span, multi-span, accounting negative), 4 FinanceBench rows (2 kept, 2 filtered) → hand-written expected pool rows
- [x] (added) `answer_type` rule table tested on one program/derivation per operation

**Dedup and splits**
- [x] Unique `item_id`; no item in two splits (S1)
- [x] No pool item has MinHash Jaccard ≥ 0.9 with any probe/test item (S1). (added) Verified with exact Jaccard on all LSH candidate pairs, not only MinHash estimates
- [x] (added) Planted exact duplicate → later one dropped; planted near-duplicate → dropped and logged; pair just below threshold → kept
- [x] Probe, fixed-demo and test sets reach target sizes after exclusions (S1)
- [x] (added) Conservation: raw count = kept + Σ excluded per reason, per dataset
- [x] (added) Stratification: each sampled set's `answer_type` counts match pool proportions up to rounding
- [x] Token budget: pool/probe/fixed_demo ≤ 2,048, test ≤ 4,096 (S1)

**Verification**
- [x] (`real_data`) FinQA executor reproduces `exe_ans` on ≥ 99% of train items; every failure logged with a reason (S1)
- [x] (`real_data`) TAT-QA evaluator verifies ≥ 97% arithmetic and ≥ 99% span items (S1)
- [x] (added) Executor unit tests per operation, constants (`const_100`, `const_m1`, …), `#k` step references and each table operation
- [x] (added) AST whitelist rejects names, calls, attributes and `__import__` strings
- [x] (added) FinanceBench items have `verified = null` and are not excluded

**Q3 checks**
- [x] Template hand-check workflow: `annotate export q3_templates` samples 50 items per template; import reports the share per template (S1)
- [ ] (human) Each template passes its 50 hand-checked items (S1)
- [x] (added) Truth-condition property: for each template item the check's direction/ordering agrees with the executed value (e.g. subtract → "increase" iff answer > 0)
- [x] (added) Zero answers and failed preconditions route to the LLM fallback
- [x] (added) With `MockStrongLLM`: schema enforced; `holds_if_answer_true = false` → one retry at 0.7 → drop and log
- [x] Validation workflow: export of 100 LLM checks stratified by dataset, import, ≥ 90% acceptance rule, `validated` column set (S1)
- [ ] (`real_data`, human) LLM checks pass human validation at ≥ 90% (S1)

**Negatives**
- [x] No `#neg` answer matches its gold under `answers_match`, incl. zero-valued and boolean cases (S1)
- [x] (added) Exactly 1,000, stratified by dataset × `answer_type`; deterministic across runs; Q3 check regenerated from the wrong answer

**Annotation CSVs** (CSV has no cell validation, so `import_sheets` enforces everything)
- [x] (added) Export → import round-trip; one CSV per annotator under `annotation/<task>/`, UTF-8, with a header row
- [x] (added) CSVs contain only the task's allowed columns (no scores, cells or validation status leak)
- [x] (added) Import rejects: a label other than `yes`/`no` (after trimming and lowercasing), a missing label, an added/removed/reordered row, or an edited item column (checked against a per-row content hash column written at export)
- [x] (added) Per-annotator row order fixed by `stable_seed(task, annotator)`; contexts with commas, quotes and newlines survive the round-trip

**Reproducibility**
- [x] Rebuilding from raw files reproduces `pool.parquet` byte for byte (S1)

---

## M3 — Jev scoring: `jev`
**Status (2026-10-23): done against `MockJev`; reworked the same day for batched requests** (Spec 1 change to cut TypeSafe cost: one request per item carrying all six questions). The real client is written to the Spec 1 request format and tested against a fake HTTP endpoint; it refuses to run until the response field names and the authentication header are taken from TypeSafe's API reference (external facts #2).
- [x] `jev_ask(state, questions)`: one `POST /v1/systemone` per item, state = `judge` rendering, six noul questions `q1 q2 q3 q1p q2p q3p`, `jev.model` pinned to `jev-1.14.0`
- [x] Cache keyed on SHA-256 of (model ID, state, canonical JSON of questions); `429` retried with backoff honouring `retry-after`; request IDs and `usage.input_tokens` logged
- [x] Tiny profile against `MockJev`
- [ ] Real Jev client run: API fields resolved 2026-10-24; the client rejects responses from any version other than `jev.model` and responses without a request-ID header. Waiting on the owner's go-ahead for the pilot's ~1,900 requests
- [x] Determinism check (200 items scored twice, batched) and batching check (same items, one question per request); mode recorded in `results/jev_manifest.json`
- [x] Outputs: `jev_scores_long` (+ `batched`, `request_id`), `jev_requests` (one row per request, input tokens), `jev_scores`
- [x] Aggregate R, I, V, indecision (+ para)

### M3 tests
- [x] Every pool item and negative has six rows with probabilities summing to 1 ± 1e-6 (S1-08)
- [x] All probabilities in [0, 1]; no pool item has null R or I (S1-09)
- [x] MockJev and the real client (on a fake endpoint): same arguments, one P(yes) per question ID, same handling of a `429` with `retry-after` (S1-16)
- [x] Exactly one six-question request per scored item in `jev_requests` (three under triplicate); answers mapped to the right question IDs (S1-17)
- [x] Batching check recorded; full run uses the selected mode, tested for both outcomes (S1-18)
- [x] (added) Request body matches the Spec 1 format exactly; the request carries the configured auth header
- [x] (added) Mock answers do not depend on batching; batched requests cost under half the input tokens of single-question requests
- [x] (added) Cache key covers model ID, state and questions (order-independent); repeats are re-sent; cached responses keep request ID and tokens
- [x] (added) Repeat rounds are sent sequentially, so request IDs are reproducible (found by the end-to-end test)
- [x] (added) Determinism-check logic tested with a mock that injects noise above and below 0.01
- [ ] (`api`) Determinism and batching checks on the real API, 200 items (S1-18)
- [x] (added) Prompt snapshot: `jev_battery.yaml` matches the Spec 1 wording character for character
- [x] (added) Aggregation on a fixture with hand-computed R, I, V, indecision; binary entropy(0.5) = 1 bit
- [x] (added) A negative's state contains the wrong answer; q3/q3p carry its regenerated check
- [x] (added) Re-running with a warm cache sends zero requests
- [x] (added) `fqsel check` verifies request accounting across `jev_scores_long` and `jev_requests`

---

## M4 — Base-model surprisal (Spec 2): `surprisal`
- [ ] `nll.answer_nll` with span logic + boundary adjustment
- [ ] vLLM `prompt_logprobs`; fp32 Transformers cross-check
- [ ] 5 reason-format samples, per-item seeds; `acc`, modal answer
- [ ] `S` = within-dataset percentile

### M4 tests
- [ ] `answer_nll` on a 3-token toy answer with known log-probs: exact sum (S2)
- [ ] 20 hand-picked answers (leading spaces, %, negatives): span covers exactly the answer tokens (S2)
- [ ] (added) Decoding the span tokens reproduces `gold_answer` (up to leading whitespace) for every tiny-profile item
- [ ] Span assertion passes or boundary adjusted and logged for 100% of items; < 1% adjusted (S2)
- [ ] (`gpu`) vLLM vs fp32 Transformers within 0.05 nats on 200 items (S2)
- [ ] Unparsed ≤ 5% overall, ≤ 10% per dataset (S2)
- [ ] (human) 50-item hand check of `extract_answer`: no errors (S2)
- [ ] Same seeds reproduce every `acc`; 20 items in a fresh process give identical text (S2)
- [ ] `S` uniform within each dataset (KS p > 0.05) (S2)
- [ ] No thinking/reasoning block in the rendered chat template (S2)
- [ ] (added) `S` is computed within dataset: permuting the other dataset's NLLs leaves it unchanged
- [ ] (added) `#neg` items are not scored; every non-derived item in all splits has one row (S2)

---

## M5 — E0 and gates (Spec 3): `comparators`, `e0`, `gates`, `frame`
- [ ] Comparators: `pvi`, `icl_reducible`, `ifd`, `judge_quality`, self-consistency subsample
- [ ] `stats/auroc.py` (paired cluster bootstrap); Spearman + Fisher-z; ICC(2,1); Cohen's κ
- [ ] A1–A6 tables; gates D1–D4; novelty flags; H1 verdict
- [ ] `analysis_frame.parquet`

### M5 tests
**Comparators**
- [ ] `pvi`, `icl_reducible`, `ifd` non-null for every pool item; `judge_quality` ≥ 99% pool and present for every `#neg` (S3)
- [ ] Self-consistency covers exactly the stored 3,000-item subsample (S3)
- [ ] (added) Each comparator's formula checked with a fake `answer_nll` returning known values (sign and direction: pvi = without − with, etc.)
- [ ] (added) `icl_reducible` uses exactly the 4 `fixed_demo` items of the item's own dataset, in fixed order

**Statistics**
- [ ] Synthetic set with AUROC 0.8 recovered within 0.02; cluster bootstrap keeps each original with its negative (S3)
- [ ] (added) AUROC equals `sklearn.roc_auc_score`; Spearman equals `scipy.stats.spearmanr`; κ equals `sklearn.cohen_kappa_score`; ICC(2,1) matches a published worked example (test tolerance 1e-9)

**Tables and gates**
- [ ] Every A1–A6 table follows the shared schema and has non-null `n` (S3)
- [ ] A1 uses exactly the 1,000 originals paired with negatives (S3)
- [ ] `gates.json` validates against `conf/gates.schema.json`; medians match a recomputation at the final `r_threshold` (S3)
- [ ] Changing `r_threshold` in D4 recomputes medians and cell counts (S3)
- [ ] (added) Gate truth table: for each of D1–D4, a metric just below and exactly at the threshold triggers correctly (`<` vs `≥` as written in Spec 3)
- [ ] (added) D4 two-step path: relax to 0.8 → still short → `cells_r_filter = false`
- [ ] H1 row matches the rule applied to the A1 and A2 tables (S3). (added) Includes the case abs ρ(I, acc) ∈ [0.5, 0.7): H1 fails, D2 not triggered

**Frame**
- [ ] One row per pool original, no derived items; `cell`/`quintile` counts match a recomputation from gates.json; `R_eff` ∈ [0, 1] (S3)
- [ ] Synthetic D1-fired gates.json: verified `arith` → `R_eff = 1`; `lookup` → `R_eff = R` (S3)
- [ ] Halted gates.json stops the `icl` stage and Specs 5–6 (S3)

---

## M6 — Proxies (Spec 4): `influence`, `icl`
- [ ] `compute_influence` (LoRA-B grads, chunked Rademacher projection, G, G_dot, grad_norm, per-dataset G, saved probe vectors)
- [ ] `run_condition` (eligible_ids or demo_plan, greedy, CBI on draws 0–9, idempotent)
- [ ] `zero_shot` pilot + timing → compute budget decision
- [ ] Note: `influence` may run on real data right after M2 (no gate dependency)

### M6 tests
**Influence (G)**
- [ ] B = 0 identity: LoRA-wrapped NLL equals unwrapped HF NLL within 1e-4 nats before any backward pass (S4) — *priority*
- [ ] Projection fidelity: projected vs unprojected cosine within 0.02 on 100 items (S4)
- [ ] SGD sign check: high-G step lowers probe loss more than low-G step (S4) — *priority*
- [ ] Per-example gradients: G alone equals G within a batch of 8 at batch size 1; test fails if accumulation is enabled (S4)
- [ ] Fresh-process recomputation of 50 items identical to 1e-5 (S4)
- [ ] `compute_influence` runs with no gates.json present (S4)
- [ ] (added) Chunked projection equals full-matrix projection on a small D (exact, same seed)
- [ ] (added) Only B matrices have nonzero gradients; model weights unchanged after the pass (checksum)
- [ ] (added) G ∈ [−1, 1]; `G_dot = G · grad_norm · ‖ḡ‖` to test tolerance; `G_finqa` uses only FinQA probes

**Demonstration runs (C)**
- [ ] Two runs of the same condition and draw → identical `demo_ids` and predictions (S4)
- [ ] Demonstrations only from the test item's own dataset, or FinQA for FinanceBench (S4)
- [ ] No demonstration is a test, probe, `fixed_demo` or derived item, unless a Spec 6 file names them (S4)
- [ ] Every prompt passes the length assertion; < 20 eligible in a dataset raises (S4)
- [ ] `run_condition` exits when `derived.halt` is true (S4)
- [ ] Tiny profile: exactly `n_draws` files, one row per test item each (S4)
- [ ] `zero_shot` pilot: ≤ 5% unparsed, nonzero accuracy per dataset (S4)
- [ ] (added) `demo_plan` used verbatim (no resampling/shuffle); a file with both or neither of `eligible_ids`/`demo_plan` is rejected
- [ ] (added) Idempotency: matching config hash → skipped; changed config → rerun
- [ ] (added) CBI rate on a hand-built example (consistent = modal share ≥ 0.8, and wrong)
- [ ] (`gpu`) Pilot timing recorded (S4)

---

## M7 — Main hypothesis tests (Spec 5): `e1`, `e2`, `figures`
- [ ] 16 condition YAMLs from the frame; supplementary composition table
- [ ] E1-G regression + joint bootstrap; simple slopes; H2/H4 under G; robustness refits
- [ ] `stats/fe_logit.py` + two-way cluster bootstrap; Holm
- [ ] E1-C (H2, H4), E2 (H3), H6; D5, D6; verdicts
- [ ] Figs 1–3
- [ ] **Then:** full tiny end-to-end + 200-item pilot → only then full runs

### M7 tests
**Conditions**
- [ ] Every condition file exists; eligible counts match a recomputation; no probe, test, `#neg` or Spec 6 item (S5)
- [ ] Cell/quintile assignment reads gates.json medians (perturbation test) (S5)
- [ ] Condition files regenerate from the frame alone; editing one frame label changes exactly the affected files (S5)
- [ ] (added) Residualization: residuals are orthogonal to acc and acc² within dataset (test tolerance)
- [ ] (added) `ifd` condition excludes ifd ≥ 1 before ranking; `judge_quality` ties broken by `stable_seed`

**Estimators**
- [ ] `fe_logit` matches statsmodels `Logit` with item dummies within 1e-4 (S5)
- [ ] (added) Fit rejected when gradient norm ≥ 1e-6; items with constant outcome dropped and counted
- [ ] (added) Bootstrap p-value formula `(1 + #reps ≤ 0)/(1 + #reps)` on a fixed vector; Holm equals `statsmodels.multipletests(method="holm")`
- [ ] (added) Simple slopes β2 + β3·q25/q75 on a hand-computed example
- [ ] Bootstrap replicates seeded and reproducible (S5)

**Simulations**
- [ ] Planted interaction (β3 = 0.2, sign-flipping slopes): sign flip recovered in ≥ 90% of 100 runs (S5)
- [ ] Planted condition effect recovered by fe_logit + two-way bootstrap; Holm over exactly 6 contrasts (S5)
- [ ] Nulls, 200 each: E1-G declares H4 in ≤ 7%; H2 C-model rejects in ≤ 7% (S5)
- [ ] (added) Planted positive quintile slope recovered under G and C; null slope ≤ 7%
- [ ] (added) H6 on simulated rankings with known ρ

**Gates, verdicts, figures**
- [ ] D5 uses only the pool-mean contrast and the H4 rule; `top_I_resid` beating pool but losing to `judge_quality` does not trigger D5 (S5)
- [ ] D5/D6 entries validate against the gates.json schema (S5)
- [ ] `hypotheses.csv` has one row per hypothesis × proxy; rerun does not duplicate (S5)
- [ ] (added) Verdict truth tables for H2, H3, H4 (incl. "positive moderation without sign flip") and overall = both proxies
- [ ] Figures regenerate from `results/` alone (S5)

---

## M8 — E3 (Spec 6): `e3`
The `e3` stage exists as a placeholder from M0 and is filled in here.
- [ ] Gate checks; battery_version match
- [ ] Base sample (hiI_hiS **and** R_eff ≥ r_threshold); low-R set; disjointness
- [ ] `#corrupt` and `#premdel` variants
- [ ] Jev manipulation check
- [ ] G contrasts; C conditions via `demo_plan`; verdicts; Fig 4

### M8 tests
- [ ] Exits on `halt`/`halt_spec6`; runs with `halt_spec6_override` set (S6)
- [ ] Every corrupted answer fails `answers_match`, keeps precision/units, absent from context (S6)
- [ ] Every premise-deleted item differs by exactly one sentence, outside support, no program argument or gold answer, still verifies (S6)
- [ ] Corrupted and premise-deleted sets share identical base items (S6)
- [ ] Every base original has `R_eff ≥ r_threshold`; no overlap with low-R, incl. with synthetic `cells_r_filter = false` (S6)
- [ ] Draw j of `e3_corrupt`/`e3_premdel` uses exactly the variants of `e3_orig`'s base items, same order (S6) — *priority*
- [ ] Every bootstrap replicate of paired contrasts carries the same multiset of draw indices across the three conditions (S6)
- [ ] Mismatched LoRA/projection seeds raise (S6)
- [ ] Variant Jev scores share battery_version and model version; cached originals not re-sent (S6)
- [ ] Planted paired ΔG (median −0.05) detected in ≥ 90% of 200 runs; null ≤ 7% (S6)
- [ ] Failed manipulation check → `manipulation_failed` regardless of contrasts (S6)
- [ ] No `e3_*` item in any Spec 5 file; Spec 5 results unchanged after Spec 6 runs (S6)
- [ ] (added) pysbd version pinned and sentence boundaries agree with FinQA indices on 50 items/dataset (`real_data`; Spec 6 open item)
- [ ] (added) Wilcoxon/Mann–Whitney match scipy; verdict truth table for H5 incl. `supported_G_only`
- [ ] (added) Low-R floor: < 20 items in both datasets → descriptive, no verdict

---

## M9 — Full runs and paper outputs
- [ ] Run all `gpu`/`api`/`real_data` tests on the GPU node; save reports to `results/test_reports/`
- [ ] Full pipeline via `dvc repro`; `python -m fqsel check` after every stage
- [ ] Commit `gates.json`, `hypotheses.csv`
- [ ] Human annotation rounds (Q3 validation, 150-item audit)
- [ ] Paper tables and figures regenerated from `results/`

### M9 tests
- [ ] (added) `fqsel check` passes on the complete set of full-run outputs
- [ ] (added) Every output's Parquet metadata names the same `model.base` revision hash and a committed git hash
- [ ] (added) Figures and paper tables rebuilt from a clean checkout of `results/` are identical to the submitted ones

---

## Blocking questions (need owner decision before dependent code)

**External facts still needed** (asked at the milestone shown)
1. Model choices: ~~`model.base`~~ resolved 2026-10-23: OLMo 3 7B Instruct at revision `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`. `model.strong` resolved 2026-10-24: Qwen3-235B-A22B-Instruct-2507 via OpenRouter, pinned to `deepinfra/fp8`. Still open: the ~0.5B tiny-profile model (M4).
2. ~~From TypeSafe's API reference: the response fields and the authentication header (`jev.api.*`).~~ *Resolved 2026-10-24* from https://docs.typesafe.ai/api/system-one and live responses: `Authorization: Bearer <key>`; answers in `answers.<qid>.noul` (P(yes), rounded to 0.01); reported model in `model`; request ID in the `x-typesafe-request-id` response header. `GET /v1/models` lists only the aliases `jev-latest` and `jev-preview`, both resolving to `jev-1.14.0`; the versioned ID `jev-1.14.0` is served and reported back when requested, and the study pins it. Still open: whether the ~260-token per-request overhead is billed as measured, and how long the versioned ID `jev-1.14.0` stays available. *(real `jev` run)*
3. ~~Raw data field names; TAT-QA test gold availability~~ — resolved 2026-10-23: confirmed against the downloaded files; TAT-QA test gold is public.

**Placeholders: values the specs leave blank** (kept as `???` until the owner decides)

| # | Placeholder | Location | Status |
| --- | --- | --- | --- |
| P1 | LoRA init seed, projection seed (my proposal: `stable_seed(20261022, "lora")` / `"proj"`) | `conf/proxy/influence.yaml` | open — asked at M6 |
| P2 | Truth condition for the `multiply` Q3 template | `fqsel/checks.py` | **resolved**: \|second factor\| vs 1; = 1 or a zero factor → LLM fallback |
| P3 | Wording of the judge-quality prompt | `prompts/judge_quality.txt` | open — asked at M5 |
| P4 | Wording of the Q3 LLM-fallback prompt | `prompts/q3_fallback.txt` | **resolved**: Claude's draft approved by the owner on 2026-10-24; file marked `STATUS: FINAL` and frozen |
| P5 | Strong-LLM temperature for self-consistency samples | `conf/strong/default.yaml` (`self_consistency`) | **resolved**: temperature 0.7, top-p 0.8, top-k 20, min-p 0, max 512 tokens, seeds `stable_seed(global_seed, item_id, "sc", k)` |
| P6 | MinHash `num_perm` and shingle size | `conf/data/default.yaml` | **resolved**: 128 permutations, word 5-grams; candidates confirmed by exact Jaccard |
| P7 | How to verify TAT-QA `count` items | `fqsel/verify/tatqa_eval.py` | **resolved**: number of `##` items equals the count and every item appears in the context |
| P8 | "Top 20%" rounding and tie-breaking for conditions other than `judge_quality` | `conf/stats/default.yaml` | open — asked at M7 |

**Resolved doc discrepancies** (see Owner decisions at the top)
- E1-C estimation: Python-only (`fe_logit` + bootstrap); no lme4.
- Human labeling: CSV files in and out; never Argilla (Spec 1 and the protocols now say so).
- `e3` stage: placeholder from M0, implemented in M8. Base items follow Spec 6 (`hiI_hiS` **and** `R_eff ≥ r_threshold`).
- Backends: HF Transformers for tiny/local tests, vLLM on the GPU node.

**Implementation choices made while building M0–M3, for owner review** (none changes a pre-registered threshold)
- *FinQA `gold_answer` formatting*: `exe_ans` at the precision shown in `qa.answer`, raised only when that precision would move the value outside the `answers_match` tolerance of `exe_ans` (about 9% of train answers disagree with `exe_ans`, e.g. `"380"` vs 3.8 → `"3.8"`). Percent answers print `exe_ans × 100` with `%`.
- *Q3 template numbers*: printed at the gold precision, raised the same way where that precision would distort the operand (otherwise a `67%` answer with operands 1.19 and 0.52 reads "The change from 1 to 1"). For `greater`, whose answer has no precision, operands use their fewest exact decimals.
- *Q3 template direction follows the given answer* for every template, so checks regenerated for wrong answers stay true under that answer. For gold answers this is identical to P2.
- *TAT-QA percent items with a top-level divide* (e.g. averages `(2.9+2.9)/2`) get "The ratio is below 100%." as Spec 1 prescribes; the template hand-check will show whether such checks are useful.
- *Exact dedup* runs over all items in `item_id` order, so a duplicate of a test item in train is dropped (test IDs sort first only by chance). Leakage removal compares pool candidates against the *sampled* probe and test items, as Spec 1 states; stratification uses the pool composition before leakage removal.
- *Split-sampling seeds* are `stable_seed(global_seed, <set name>, <dataset>, <stratum>)`, e.g. `(20261022, "probe", "finqa", "arith")`; negatives use `(global_seed, "negatives", "<dataset>|<answer_type>")`; the determinism sample `(global_seed, "jev_determinism")`; LLM fallback calls `(global_seed, item_id, "q3", attempt)`.
- *First Q3 fallback attempt is greedy* (temperature 0); Spec 1 fixes only the retry temperature (0.7).
- *Items whose Q3 fallback fails twice* stay in `pool.parquet`, are listed in `data/q3_dropped.csv`, get no Jev scores, and must be excluded by the analysis frame (M5).
- *`#neg` copies of span items* still pass verification, because span verification only checks presence in the context; Spec 3 uses `verified` for originals only.
- *Pilot profile* uses 100 pool items per dataset, my reading of "200-item pilot".
- *Parquet metadata* carries no timestamp, so `pool.parquet` rebuilds byte for byte (Spec 1); `jev_scores_long` keeps its spec'd `timestamp` column.

**Findings that affect later milestones**
- *FinanceBench*: only 52 of 150 open-source items have a single-number gold answer with evidence, before the 4,096-token budget. Spec 4 hypothesis-tests out-of-distribution results only if ≥ 50 survive, so this is likely to end up descriptive.
- *Token budget with the toy tokenizer*: the TEST-ONLY toy tokenizer (600-token vocabulary) overcounts heavily, excluding 4,840 of 6,251 FinQA train items under the 2,048 budget. Real exclusion counts must come from the `model.base` tokenizer.
- *FinQA lookup items are rare*: 29 of 1,399 surviving FinQA train items in the toy-tokenizer run (185 of 6,251 overall end in a table operation). Stratified sampling handles this, but per-dataset `lookup` analyses will be thin for FinQA.

---

## Verification log

Real-data, GPU, API and human checks, recorded as they are run. One row per check; never overwrite a row: add a new one when rerun.

| Date | Commit | Check (test ID) | Profile | Result | Pass | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 2026-10-23 | no-commit | MANIFEST hashes (S1-01) | — | 8/8 files match | yes | `pytest -m real_data` |
| 2026-10-23 | no-commit | FinQA executor on train (S1-04) | — | 6,251/6,251 = 100% reproduce `exe_ans` | yes | target ≥ 99% |
| 2026-10-23 | no-commit | TAT-QA evaluator on train (S1-05) | — | arithmetic 99.33% (5,506/5,543), spans 99.99% (7,366/7,367), count 100% (305/305) | yes | targets ≥ 97% / ≥ 99%; with case-insensitive presence and accounting negatives (owner decisions) |
| 2026-10-23 | no-commit | TAT-QA evaluator on dev / test_gold | — | arith 99.30% / 98.28%; spans 99.89% / 100%; count 100% / 100% | — | informational |
| 2026-10-23 | no-commit | Span verification, strict case-sensitive (before decision) | — | 97.98% of train spans | no | led to the case-insensitive owner decision |
| 2026-10-23 | no-commit | All real items fit the pool schema | — | 24,879 items valid; FinanceBench kept 52, filtered 98 | yes | |
| 2026-10-23 | no-commit | Tiny pipeline on real data: pool → verify → q3 → jev → check | tiny | 248 rows (20 negatives), 140 Q3 checks (98 template, 42 LLM), 840 Jev rows; `check: OK` | yes | toy tokenizer, mock clients |
| 2026-10-23 | no-commit | Exact leakage check on the tiny real-data build (S1-03) | tiny | no pool item ≥ 0.9 Jaccard with probe/test | yes | inverted shingle index, independent of MinHash |
| 2026-10-23 | no-commit | Full test suite | tiny | 325 passed (6 real_data/gpu deselected); ruff clean | yes | `pytest -q` |
| 2026-10-23 | 53ef30d+ | Batched Jev stage on the tiny real-data build | tiny | 140 items → 140 six-question requests; batching check: batched = single (mode batched); `check: OK` | yes | MockJev; token counts are mock estimates |
| 2026-10-23 | 53ef30d+ | Full test suite after the batching change | tiny | 332 passed; ruff clean; e2e and Jev tests stable over 8 repeats | yes | |
| 2026-10-23 | e2a3ce8+ | OLMo 3 7B Instruct chat template (S2-09) | — | no reasoning block in direct or reason format; default system prompt present | yes | `real_data`, tokenizer files only |
| 2026-10-23 | e2a3ce8+ | Token budget with the model.base tokenizer | — | over budget: FinQA train 0.78% (49), TAT-QA train 0.36% (48), TAT-QA dev 0.72% (12), all other splits 0%; FinanceBench 52/52 fit | yes | max qa_tokens 3,925 (TAT-QA train) |
| 2026-10-24 | b8e73a0+ | Pilot pool and verify on real data (after two fixes: pilot negatives 1,000 → 100; multi-span negatives with a placeholder first span) | pilot | pool 1,860 rows (200 pool, 100 negatives), 245 exclusions; verify 1,722/1,860 | yes | negatives fail verification by construction |
| 2026-10-24 | b8e73a0+ | Pilot q3 with real Qwen3-235B via OpenRouter | pilot | 299 checks (213 template, 86 LLM), 1 negative dropped after two failed fallbacks; 89 calls, all served by DeepInfra, $0.009 | yes | annotation CSVs exported for q3_validation (86 rows) and q3_templates (120 rows) |
| 2026-10-24 | b8e73a0+ | Live TypeSafe probes (GET /v1/models; 3 tiny POST /v1/systemone) | — | response shape matches the API reference; `jev-1.13.0` served and reported; `jev-latest` → `jev-1.14.0`; unknown ID → 400; noul rounded to 0.01 | yes | basis for the `jev.api.*` values |
| 2026-10-24 | b8e73a0+ | Pilot jev dry run with MockJev (request count) | pilot | preflight 1,600 requests (~1.81M input tokens); scoring 299 requests (~0.37M) if batched and deterministic | — | MockJev token counts are estimates (chars/4 + 260) |
| 2026-10-24 | b8e73a0+ | Full test suite | tiny | 351 passed; ruff clean | yes | |

## Owner decisions log

| Date | Item | Decision |
| --- | --- | --- |
| 2026-10-23 | Project rules R1/R2 | Removed from Spec 1; no banned-name test needed |
| 2026-10-23 | Blank spec values | Keep as placeholders; ask owner during implementation |
| 2026-10-23 | Statistics stack | Python only |
| 2026-10-23 | Human labeling | CSV out, CSV back; never Argilla |
| 2026-10-23 | E3 | Placeholder stage until M8 |
| 2026-10-23 | Backends | HF Transformers for tiny tests, vLLM for GPU |
| 2026-10-23 | P2 multiply template | \|second factor\| vs 1; = 1 or zero factor → LLM fallback |
| 2026-10-23 | P6 MinHash | 128 permutations, lowercased word 5-grams |
| 2026-10-23 | P7 TAT-QA count | `##` item count equals the gold count and every item appears in the context |
| 2026-10-23 | P4 Q3 fallback prompt | Claude drafts it; owner approves before real runs |
| 2026-10-23 | Span swap for negatives | Same word count; non-numeric table cells first, then paragraph n-grams; one span replaced for multi-span |
| 2026-10-23 | `render_item` layout | "Context:" / "Table:" / "Question:" / "Stated answer:" blocks; FinQA pre-text before the table, post-text after |
| 2026-10-23 | Span and count presence check | Case-insensitive, whitespace- and curly-quote-normalised (strict matching gave 97.98% vs the 99% target) |
| 2026-10-23 | TAT-QA accounting negatives | A parenthesised bare number in a derivation is negative, "(6,919)" → −6919 |
| 2026-10-23 | Table-op Q3 templates | Templates win for any program ending in a templated op, including FinQA table-op (lookup) items |
| 2026-10-23 | Jev cost reduction (Spec 1 edit) | One batched request per item with all six questions; `jev.model` pinned to `jev-1.13.0`; batching verified against single-question requests on 200 items |
| 2026-10-23 | Jev model version | `jev-1.13.0` confirmed for the whole study (no later version); fixed before any real scoring |
| 2026-10-23 | `model.base` | OLMo 3 7B Instruct (`allenai/Olmo-3-7B-Instruct`, revision `6e5971d9eba42665f5bd5a0fcf047f299ce1dccc`): fully open training data allow a contamination audit; chat template used unchanged (it inserts a default system prompt); Llama 3.1 8B Instruct suggested as an exploratory H4 robustness check |
| 2026-10-24 | `model.strong` | Qwen3-235B-A22B-Instruct-2507 via OpenRouter (`qwen/qwen3-235b-a22b-2507`), provider endpoint pinned to `deepinfra/fp8`, no fallbacks, `require_parameters`; chosen over Claude Sonnet 5.5 (~$62–125, no temperature control) and smaller Qwen sizes (weaker H2 judge and A4 reference); estimated cost ~$6, budget $15–25. Departs from the protocol's "frontier API" wording |
| 2026-10-24 | `model.strong` licence | Qwen3-235B-A22B-Instruct-2507 is Apache 2.0 (https://huggingface.co/Qwen/Qwen3-235B-A22B-Instruct-2507/blob/main/LICENSE; model card revision `ac9c66cc`) |
| 2026-10-24 | `model.base` licence | OLMo 3 7B Instruct is Apache 2.0, as declared on its model card (https://huggingface.co/allenai/Olmo-3-7B-Instruct) at the pinned revision `6e5971d9`; no separate LICENSE file in the repository |
| 2026-10-24 | P5 self-consistency sampling | Temperature 0.7 (matches Spec 2 accuracy and Spec 4 CBI sampling; Qwen's recommended value), top-p 0.8, top-k 20, min-p 0 (Qwen's recommended settings), max 512 tokens (Spec 2 reason format), 5 separate requests with seeds `stable_seed(global_seed, item_id, "sc", k)` |
| 2026-10-24 | P4 Q3 fallback prompt approved | Owner set `prompts/q3_fallback.txt` to `STATUS: FINAL`; wording frozen (a change requires regenerating all LLM checks) |
| 2026-10-24 | Jev API fields (`jev.api.*`) | From https://docs.typesafe.ai/api/system-one and live responses: `Authorization: Bearer <key>`; answers in `answers.<qid>.noul` (P(yes), rounded to 0.01); reported model in `model`; request ID in the `x-typesafe-request-id` response header. `GET /v1/models` lists only the aliases `jev-latest` and `jev-preview`, both resolving to `jev-1.14.0`; the versioned ID `jev-1.14.0` is served and reported back when requested, and the study pins it. The real client now rejects any response whose `model` is not `jev.model`. |
| 2026-10-24 | Placeholder check per stage | Spec 1 stages and `annotate` ignore P1, P3 and P8 (`Stage.ignores`); reading an ignored `???` still raises, and `load_prompt` still refuses non-FINAL prompts |
| 2026-10-24 | Pilot negatives | 100 (pilot only; the full run keeps Spec 1's 1,000): a 200-item pool cannot supply 1,000 negatives |
| 2026-10-24 | Jev model version changed to `jev-1.14.0` | Owner decision: every experiment uses `jev-1.14.0`, and `jev-1.13.0` is not used anywhere. Supersedes the 2026-10-23 rows above. Both aliases (`jev-latest`, `jev-preview`) resolve to `jev-1.14.0`, but the config names the versioned ID and the client refuses aliases. No real Jev scoring had run under `jev-1.13.0` |
