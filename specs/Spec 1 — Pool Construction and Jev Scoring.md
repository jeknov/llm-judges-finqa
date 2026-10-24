# Spec 1 — Pool Construction and Jev Scoring

Oct 22, 2026 · @Jekaterina

## Purpose

This spec builds the item pool and produces every per-item score that comes from data or Jev: verified correctness, Q3 consequence checks, and the reliability R and implication strength I. It is Step 1 of protocol v1 and runs first; Specs 2–6 read its outputs and never call Jev again, except Spec 6 for its manipulation check.

Outputs consumed downstream: `pool.parquet` (all specs), `verified.parquet` (Specs 3 and 6), `q3_checks.parquet` (this spec and Spec 6), and `jev_scores.parquet` (Specs 3, 5 and 6).

## Shared conventions

These conventions apply to all six specs; other specs reference this section rather than restating it.

**Item IDs.** Every item has `item_id = "<dataset>:<source_id>"`, where dataset is `finqa`, `tatqa` or `financebench`. For FinQA, `source_id` is the record `id`. For TAT-QA, it is the question `uid`, since one TAT-QA record holds several questions. For FinanceBench, it is the release's question ID. Derived items append a suffix: `#neg` for validation negatives (this spec), and `#corrupt` or `#premdel` for manipulated items (Spec 6). A derived item's base ID is everything before `#`.

**Splits.** Each item carries `split ∈ {pool, probe, test, fixed_demo}`. Pool items come from the source train splits; probe items and the 8 fixed demonstrations from the dev splits; test items from the test splits and FinanceBench. No item ever changes split. Derived items inherit their base item's split, but are excluded from every pool-level analysis by `is_derived = true`.

**File layout.**

| Path | Written by | Contents |
| --- | --- | --- |
| `data/raw/` | manual download | Source files, unmodified, hashes in `data/raw/MANIFEST` |
| `data/pool.parquet` | Spec 1 | One row per item, all splits, including `#neg` items |
| `data/verified.parquet` | Spec 1 | Execution result per item |
| `data/q3_checks.parquet` | Spec 1 | One Q3 check per item |
| `scores/jev_scores_long.parquet` | Spec 1 | Jev probabilities per item × battery × question |
| `scores/jev_scores.parquet` | Spec 1 | R, I, V and indecision per item |
| `scores/surprisal.parquet` | Spec 2 | NLL and sampled accuracy per item |
| `runs/surprisal/samples.parquet` | Spec 2 | Raw sampled generations |
| `scores/comparators.parquet` | Spec 3 | PVI, ICL-reducible loss, IFD, judge quality, self-consistency |
| `results/e0/` | Spec 3 | Tables A1–A6 |
| `results/gates.json` | Specs 3 and 5 | Gate decisions and derived values |
| `data/analysis_frame.parquet` | Spec 3 | One row per pool item with every score and gate-derived label |
| `scores/influence.parquet`, `grads/` | Spec 4 | G per item; projected gradients |
| `conf/conditions/` | Specs 5 and 6 | One YAML file per C condition |
| `runs/icl/<condition>/<draw>.parquet` | Spec 4 | C generations |
| `results/e1g/`, `results/e1c/`, `results/e2/`, `results/h6/` | Spec 5 | Analysis tables |
| `results/hypotheses.csv` | Specs 3, 5, 6 | One verdict row per hypothesis × proxy |

Spec 6 adds `data/e3_items.parquet` (manipulated variants, pool schema plus three columns), `scores/jev_scores_e3.parquet`, `scores/surprisal_e3.parquet` and `results/e3/`. `read_table("items")` returns the union of `pool.parquet` and `e3_items.parquet`, which is how every engine looks items up.

**Config.** All parameters live in Hydra configs under `conf/`; no constant in code. The base-model checkpoint is one config key, `model.base`, used by Specs 2–6 unchanged.

**Determinism.** Global seed 20261022. Every sampled quantity records its seed. Every output file records the git commit, config hash and model identifier in Parquet metadata. Outputs are tracked with DVC.

**Shared helpers.** Each function below exists once, in the module named, and every spec calls it rather than reimplementing it.

| Function | Module | Used by | Contract |
| --- | --- | --- | --- |
| `render_item(item, mode)` | `fqsel/prompts.py` | all | `qa`: context and question. `judge`: context, question and stated answer. Tables as pipe-delimited rows with a header, numbers as in the source. |
| `build_messages(item, fmt, demos=[])` | `fqsel/prompts.py` | Specs 2–6 | Chat messages for `fmt` ∈ {direct, reason}, with demonstrations as prior user and assistant turns in the `direct` format. The only place prompts are assembled. |
| `extract_answer(text, answer_type)` | `fqsel/prompts.py` | Specs 2–6 | Defined in Spec 2. |
| `answers_match(pred, gold, answer_type)` | `fqsel/match.py` | Specs 1–6 | Defined in this spec. |
| `answer_nll(items, fmt, demos)` | `fqsel/nll.py` | Specs 2, 3, 6 | Teacher-forced answer NLL, defined in Spec 2. |
| `stable_seed(*parts)` | `fqsel/ids.py` | all | `int(sha256("\|".join(str(p) for p in parts)).hexdigest()[:16], 16)`. Process-independent, so the built-in `hash` is never used. |
| `read_table(name)`, `write_table(name, df)` | `fqsel/io.py` | all | Validate against the schema for `name` on every read and write, and stamp the metadata described under Determinism. |
| `record_verdict(hypothesis, proxy, verdict, rule, evidence)` | `fqsel/verdicts.py` | Specs 3, 5, 6 | Upserts one row of `results/hypotheses.csv`, keyed by hypothesis and proxy. |

Two further helpers are shared between this spec and Spec 6: `make_wrong_answer(item, tag)` in `fqsel/perturb.py`, which builds the `#neg` answers here and the `#corrupt` answers there, with the tag entering the seed; and `generate_check(item, answer=None)` in `fqsel/checks.py`, which generates the Q3 check for the given answer, defaulting to the gold. `export_sheets(task, items)` and `import_sheets(task)` in `fqsel/annotate.py` are shared with Spec 3 for all human labeling, which uses CSV files sent to annotators and returned with labels (see Q3 check generation, Annotation sheets).

**Token budget.** Item token counts use the `model.base` tokenizer and the `qa` rendering. Pool, probe and fixed-demo items above 2,048 tokens and test items above 4,096 tokens are excluded in Pool construction. The base model must have a context of at least 16,384 tokens, which guarantees that 4 demonstrations plus any test item always fit, so no spec needs to shrink k.

**Code layout.** One Python package, `fqsel`, with one module per stage and a single command, `python -m fqsel run <stage>`. The stages are, in order: `pool`, `verify`, `q3`, `jev`, `surprisal`, `comparators`, `e0`, `gates`, `frame`, `influence`, `icl`, `e1`, `e2`, `figures`. Each stage is one DVC stage whose dependencies are the tables it reads, so `dvc repro` reruns only what changed. Table schemas are pandera models in `fqsel/schemas.py`, one per file in the layout table.

**Testing strategy.** Four layers of tests, all in CI:

1. Unit tests for every shared helper, with the hand-written suites each spec lists, and snapshot tests that freeze the exact text `build_messages` produces for 10 fixed items in each format, so that any prompt change is visible in review.
2. Schema and invariant tests that run on every write: unique IDs, split disjointness, no probe, test, fixed-demo or derived item in any selection list, and probabilities in \[0, 1\].
3. Statistical tests on simulated data: each analysis recovers a planted effect at the stated rate, and on simulated null data its false-positive rate stays at or below 7% over 200 simulations.
4. An end-to-end smoke test, `profile=tiny`: 60 pool, 20 probe and 20 test items per dataset, a small open model of about 0.5B parameters, and `MockJev` and `MockStrongLLM` clients that return seeded, deterministic outputs through the real interfaces. It runs every stage in order in under 20 minutes, checks that every output validates and that `gates.json` is well formed, and is run twice to confirm byte-identical outputs apart from timestamps.

**Deviations from protocol v1.** The specs refine the protocol in these places; the protocol text is otherwise authoritative, and the paper reports the spec versions.

| Topic | Protocol v1 | Specs | Reason |
| --- | --- | --- | --- |
| Reliability validation | AUROC of R against verified correctness | AUROC on 1,000 originals plus 1,000 `#neg` copies with wrong answers | Execution verifies nearly every gold label, so there would be no negatives |
| Surprisal S | Mean token NLL | Within-dataset percentile of mean token NLL | Answer length and number format differ between datasets |
| Proxy C format | Unspecified | `direct` format, greedy decoding; CBI from 5 samples on the first 10 draws | A demonstration is exactly its question and answer; caps sampling cost |
| Demonstrations k | 4, reduced to 2 when too long | Always 4, guaranteed by the token budget | Removes a per-dataset special case |
| Self-consistency | Strong LLM, 5 samples | Same, on a 3,000-item stratified subsample | API cost |
| C conditions | About 14 | 16, adding `zero_shot` and counting quintiles separately | Explicit baseline |
| Mixed models | `(1 \| test_item) + (1 \| draw)` | `(1 \| test_item) + (1 \| condition:draw)` in Spec 5; crossed `(1 \| draw)` for Spec 6's three paired conditions | Draws of different conditions share no demonstrations, so the draw term is nested in condition; Spec 6's paired conditions do share draws by construction |
| Mixed-model estimation | Random intercepts for test item and draw | Test-item fixed effects in a logistic regression, with a two-way cluster bootstrap over test items and draws in place of the draw random effect (Spec 5, E1-C) | The pipeline is Python-only, and no Python library fits crossed random effects in a logistic model reliably at this scale; with hundreds of rows per item, the incidental-parameter bias of item fixed effects is negligible |
| E1-G uncertainty | Bootstrap over probes | Joint bootstrap over probes and pool items | Includes item sampling error |
| Gate D4 | Any cell below 200 items | Any dataset × cell below 100 items; then relax R to 0.8, then drop the R filter | Demonstrations are drawn per dataset |
| Out-of-distribution test | FinanceBench subset | Same, hypothesis-tested only if at least 50 items survive filtering | Too few items for inference otherwise |

## Pool construction

The pool is FinQA train plus TAT-QA train after deduplication and leakage removal; probe and test sets come from dev and test. Field names below are from memory of the public releases and must be checked against the downloaded files before coding.

**Field mapping to `pool.parquet`.**

| Column | Type | FinQA source | TAT-QA source |
| --- | --- | --- | --- |
| `item_id` | str | `finqa:` + `id` | `tatqa:` + question `uid` |
| `dataset` | str | `finqa` | `tatqa` |
| `split` | str | from file | from file |
| `question` | str | `qa.question` | question `question` |
| `context_text` | list\[str\] | `pre_text` + `post_text`, sentence order kept, with a `pos` marker for pre/post | `paragraphs[].text` in `order` |
| `table` | list\[list\[str\]\] | `table` | `table.table` |
| `program` | str or null | `qa.program` | null |
| `derivation` | str or null | null | `derivation` |
| `gold_answer` | str | `qa.exe_ans` formatted to the precision in `qa.answer` | `answer` joined with `scale` |
| `gold_value` | float, list or str | `qa.exe_ans` | numeric `answer` × scale, or span list |
| `answer_type` | str | `arith` if the program contains add, subtract, multiply, divide, exp or greater; else `lookup` | `arith` for arithmetic and count; `lookup` for span and multi-span |
| `support_ids` | list\[str\] | `qa.gold_inds` keys | `rel_paragraphs` plus table if `answer_from` includes table |
| `context_chars` | int | computed | computed |

Three columns are added for every item: `qa_tokens` (int, the `qa` rendering's length in `model.base` tokens), `is_derived` (bool) and `base_item_id` (str, equal to `item_id` for originals).

**Deduplication.**

1. Exact: drop items whose normalized question plus context hash matches an earlier item; keep the first by `item_id` order.
2. Leakage: drop pool items with MinHash Jaccard ≥ 0.9 on question plus context against any probe or test item. Log every dropped pair to `data/dedup_log.csv`.

**Probe and test sets.**

- Probe: 500 items from the dev splits, 250 per dataset, stratified by `answer_type` in proportion to the pool.
- Fixed demonstrations: 4 further dev items per dataset, stratified the same way and disjoint from the probe, with `split = fixed_demo`. Spec 3 uses them for the ICL-reducible comparator.
- Test: 500 items per dataset from the test splits, stratified the same way. If TAT-QA test gold answers are not public, take TAT-QA test items from dev instead, disjoint from the probe and fixed demonstrations, and record this in the manifest.
- FinanceBench: from the public open-source release, keep questions whose gold answer `answers_match` can parse as a single number with optional units, and which come with evidence text. The evidence text becomes the context. These items get `split = test`, `answer_type = arith` and `dataset = financebench`.

**Exclusions.** Drop items with an empty gold answer, items that fail verification (see Verified correctness), and items over the token budget: `qa_tokens` above 2,048 for pool, probe and fixed-demo items, and above 4,096 for test items. Exclusions are applied before probe, test and fixed-demo sampling, so that sampled sets reach their target sizes. Report counts per reason and dataset.

## Verified correctness

Execution confirms that each gold answer follows from its program or derivation; it does not detect labels whose program is itself wrong. Because FinQA's `exe_ans` is the program output by construction, nearly every pool item will verify, so the AUROC in Spec 3 needs constructed negatives. They are built and scored in this spec (see Jev battery scoring), so that every Jev call except Spec 6's sits in one place.

**FinQA executor.** Implement the FinQA program DSL: `add`, `subtract`, `multiply`, `divide`, `exp`, `greater`, and `table_sum`, `table_average`, `table_max`, `table_min` over a named table row. Arguments are numbers, constants (`const_100`, `const_1000`, `const_m1` and the rest of the released constant list), step references `#k`, and table row names. Port the reference executor from the official FinQA repository rather than rewriting it, and test it against every train item.

**TAT-QA evaluator.** For arithmetic and count items, parse `derivation` as an arithmetic expression with an AST whitelist (numbers, + − × ÷, parentheses, unary minus); never call `eval`. Strip thousands separators and percent signs before parsing. For span and multi-span items, verification means every gold span appears verbatim, after whitespace normalization, in the paragraphs or table cells.

**Comparison function `answers_match(pred, gold, answer_type)`.** Used here, in Spec 2 for accuracy and in Spec 4 for scoring.

- Numbers: parse both sides, accepting percent and scale words; a percentage matches either its fraction or its percent form. Match if the relative error is ≤ 0.5% or the absolute error is ≤ 0.005 after rounding both to the gold precision.
- Spans: case-insensitive exact match after removing punctuation and articles; multi-span requires set equality.
- Booleans from `greater`: map yes/no and true/false to one form.

**Output `verified.parquet`.** Columns: `item_id`, `exec_value`, `verified` (bool), `failure_reason` (null, `parse`, `exec_error`, `mismatch`, `span_missing`). Items with `verified = false` are excluded and listed by reason. FinanceBench items have no program; they are stored with `verified = null`, which does not exclude them.

## Q3 check generation

Every pool item and validation negative gets exactly one check (probe and test items need none): a short statement that is true whenever the gold answer is true. Q3 then asks Jev whether the answer entails it, so the "yes" probability measures how visibly the answer forces a consequence. Checks are generated from the program or derivation where possible and by a strong LLM otherwise.

**Deterministic templates.** The template is chosen by the final operation of the FinQA program or the top-level operator of the TAT-QA derivation. Numbers in the check come from the executed values.

| Final operation | Check type | Template | Truth condition |
| --- | --- | --- | --- |
| subtract | direction | "The change from {b} to {a} is an {increase / decrease}." | sign of the answer |
| divide | ordering | "The ratio is {above / below} 1." (percent items: "above / below 100%") | answer vs 1 |
| add, table\_sum | total | "The total is larger than {largest addend}." | all addends positive; else fall back to LLM |
| multiply | magnitude | "The result is {larger / smaller} in absolute value than {first factor}." |  |
| greater | ordering | "{first value} is {larger / not larger} than {second value}." | the boolean answer |
| table\_average | total | "The average lies between {row min} and {row max}." | always true |
| table\_max, table\_min | ordering | "No other value in the row is {larger / smaller} than the answer." | always true |

{a} and {b} are the numeric arguments, printed with the gold precision. When the answer is exactly zero or a template's precondition fails, the item goes to the LLM fallback.

**LLM fallback.** Used for lookup items and template failures. The strong LLM (config key `model.strong`) receives the `judge` rendering and must return JSON with the schema `{check_text: str, check_type: direction | total | ordering | magnitude | unit, holds_if_answer_true: bool}`, enforced with the API's structured-output mode. Instructions: the check must follow from the answer together with the context, must not restate the answer, and must be decidable from the context. Reject outputs with `holds_if_answer_true = false` and retry once at temperature 0.7; if the retry also fails, drop the item and log it.

**Annotation sheets.** Human labeling here and in Spec 3's audit uses plain CSV files, never an annotation platform such as Argilla, through one module, `fqsel/annotate.py`. `export_sheets(task, items)` writes one CSV file per annotator under `annotation/<task>/`, UTF-8 with a header row and standard quoting, so that contexts containing commas, quotes or line breaks survive. Each file holds the item columns, a `row_hash` of those columns per row, empty label columns to be filled with `yes` or `no`, an optional note column, and rows in a per-annotator order from `stable_seed(task, annotator)`. Files contain only the fields the task names, so scores, cell membership and validation status cannot leak. Each annotator receives only their own file, fills in the label columns, and returns the same CSV file with the labels. Because a CSV cannot restrict cell values, `import_sheets(task)` enforces them: it reads the returned files, accepts `yes` or `no` after trimming and lowercasing, rejects any file with a missing or invalid label, an added, removed or reordered row, or an item column that no longer matches its `row_hash`, and writes the labels to a Parquet table with annotator, item and timestamp.

**Validation.** Draw 100 LLM-generated checks, stratified by dataset, for two annotators, exported as annotation sheets, labeling each "follows from the answer" yes or no. Accept the generator if at least 90% are labeled yes by both; otherwise revise the prompt and regenerate all LLM checks. Template checks are validated by unit tests on 50 hand-checked items per template.

**Output `q3_checks.parquet`.** Columns: `item_id`, `check_text`, `check_type`, `source` (template or llm), `template_op` (null for llm), `generator_model`, `validated` (bool, for sampled items).

## Jev battery scoring

Jev scores every pool item and every validation negative on six questions: the battery Q1–Q3 and its paraphrase Q1′–Q3′. R and I are computed from the stored probabilities; nothing downstream calls Jev again except Spec 6.

**Client interface.** Wrap Jev behind one function, `jev_ask(state: str, questions: dict[str, str]) -> dict[str, float]`, which sends one request to `POST https://api.typesafe.ai/v1/systemone` with body `{"model": "<jev.model>", "state": state, "questions": {qid: {"type": "noul", "instructions": text}, ...}}` and returns P(yes) for each question ID. Pin `jev.model` to a versioned ID (`jev-1.13.0`), never an alias, because aliases move to new releases and would change scores mid-study; record the versioned ID each response reports. Each item is scored in a single request carrying all six battery questions: Jev ingests the state once and answers every question against it in parallel, and the questions cannot see one another's answers, so batching does not couple them. Billing is per input token with output free, so one batched request costs roughly the state once plus six short questions, instead of the state six times. The wrapper caches responses keyed by the SHA-256 of `(model ID, state, canonical JSON of questions)`, retries `429` responses with exponential backoff that honours `retry-after`, and logs request IDs and `usage.input_tokens`. The response field holding a noul answer's probability must be taken from TypeSafe's API reference; nothing here assumes its name.

**State and questions.** The state is the `judge` rendering of the item (context, question and stated answer), identical for all six questions. Each question below is sent as a noul question whose `instructions` are the wording in the table, with `{check_text}` filled from `q3_checks.parquet`. Question IDs are `q1`, `q2`, `q3`, `q1p`, `q2p`, `q3p`. P(no) is stored as 1 − P(yes). The wording is frozen before any score is looked at; changing it invalidates all scores.

| Question | Construct | Wording |
| --- | --- | --- |
| Q1 | support | Given the context and question, is the stated answer supported by the context? |
| Q2 | determinacy | Do the context and question determine this answer uniquely, such that no materially different answer would also be defensible? |
| Q3 | consequence | Assume the stated answer is true. Does it entail the following check? Check: {check\_text} |
| Q1′ | support | Does the information provided back up the answer given to this question? |
| Q2′ | determinacy | Is this the only reasonable answer to the question given the information, rather than one of several defensible answers? |
| Q3′ | consequence | If the answer given is correct, must the following statement also be correct? Statement: {check\_text} |

**Validation negatives.** Draw 1,000 pool items stratified by dataset and `answer_type`. For each, create a copy with `item_id` suffix `#neg` and a wrong gold answer:

- Numeric: multiply the gold value by a factor drawn from {0.5, 0.8, 1.25, 2} with `stable_seed(item_id, "neg")`, or flip its sign for direction-type items, keeping the gold precision and units. If the gold value is 0, use another numeric value from the same table instead.
- Boolean answers from `greater`: flip yes and no.
- Span: substitute a different span of the same length class from the same context.
- Confirm with `answers_match` that the new answer does not match the gold; redraw up to 5 times, then drop the item and draw a replacement.

Negatives get their Q3 check regenerated from the wrong answer with the same rules, so that only Q1 should separate them from originals. They are scored on all six questions, used only in Spec 3, and never enter any selection condition.

**Determinism and batching checks.** Before the full run, score 200 items twice in batched requests. If the mean absolute difference in P(yes) exceeds 0.01, score every item three times and store the mean. On the same 200 items, also score each question in its own single-question request, and confirm that batched and single-question probabilities differ by no more than the repeat difference plus 0.01. If they differ more, score all items one question per request and record the decision in the manifest.

**Aggregation.** For each battery b ∈ {orig, para}:

- R\_b = P(yes | Q1\_b)
- I\_b = mean of P(yes | Q2\_b) and P(yes | Q3\_b)
- V\_b = R\_b × I\_b
- indecision\_b = the maximum over the three questions of the binary entropy of P(yes), in bits

The original battery is primary; the paraphrase is used only for robustness analyses in Spec 3.

**Outputs.** `jev_scores_long.parquet`, one row per item × battery × question: `item_id`, `battery`, `question`, `p_yes`, `p_no`, `entropy_bits`, `n_calls`, `batched` (bool), `jev_model_version`, `request_id`, `timestamp`. `jev_requests.parquet`, one row per request: `request_id`, `item_id`, `n_questions`, `input_tokens`, `jev_model_version`, `timestamp`, used for cost tracking. `jev_scores.parquet`, one row per item: `item_id`, `is_negative`, `R`, `I`, `V`, `indecision`, `R_para`, `I_para`, `V_para`, `indecision_para`.

## Acceptance tests and open items

Spec 1 is done when every test below passes in CI and the open items are resolved or logged.

- [ ] Raw file hashes in `data/raw/MANIFEST` match the downloaded files.
- [ ] Every `item_id` is unique, and no item appears in two splits.
- [ ] No pool item has MinHash Jaccard ≥ 0.9 with any probe or test item.
- [ ] The FinQA executor reproduces `exe_ans` on ≥ 99% of train items; every failure is logged with a reason.
- [ ] The TAT-QA evaluator verifies ≥ 97% of arithmetic items and ≥ 99% of span items.
- [ ] `answers_match` passes a hand-written suite of 100 cases covering percent forms, scale words, rounding and multi-span.
- [ ] Each template passes its 50 hand-checked items; the LLM checks pass human validation at ≥ 90%.
- [ ] Every pool item and every negative has six Jev rows with probabilities summing to 1 ± 1e-6.
- [ ] All probabilities in `jev_scores.parquet` lie in \[0, 1\], and no pool item has a null R or I.
- [ ] Re-running the pipeline from raw files with the cache cleared reproduces `pool.parquet` byte for byte, and Jev scores within the determinism tolerance.

* [ ] Every pool, probe and fixed-demo item has `qa_tokens` ≤ 2,048, and every test item ≤ 4,096.
* [ ] Probe, fixed-demo and test sets reach their target sizes after exclusions.
* [ ] `stable_seed` returns the same value in two separate processes, and no module calls the built-in `hash` (enforced by a lint rule).
* [ ] Snapshot tests of `build_messages` pass for 10 items in each format.
* [ ] No `#neg` answer matches its gold under `answers_match`, including the zero-valued and boolean cases.
* [ ] `MockJev` and the real client pass the same interface test: identical argument types, one P(yes) in [0, 1] per question ID, and identical handling of a `429` with `retry-after`.
* [ ] Every scored item has exactly one batched request with six questions in `jev_requests.parquet` (three if the determinism check triggered repeats), and the six answers map to the right question IDs.
* [ ] The batched-versus-single-question check on 200 items is recorded, and the full run used the mode it selected.

**Open items.**

- Confirm the FinQA and TAT-QA field names against the downloaded files, and whether TAT-QA test gold answers are public.
- Confirm in TypeSafe's API reference the response field that holds a noul answer's probability, and whether the per-request token overhead is billed as measured externally (about 260 tokens); pin `jev.model` to `jev-1.13.0` or a later versioned ID chosen before scoring starts.
- Choose `model.strong` for Q3 fallback checks; it must support structured output.
