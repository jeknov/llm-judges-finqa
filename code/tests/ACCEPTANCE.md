# Acceptance traceability matrix

Every item in the specs' "Acceptance tests" lists has an ID here. A test covering an item carries
`@pytest.mark.spec("<ID>")`. `tests/unit/test_guardrails.py::test_acceptance_traceability` fails if an
`implemented` or `partial` ID has no test, or if a test names an ID that is not in this table.

Status: `implemented` (fully covered by automated tests), `partial` (automated part covered; a human,
GPU, API or later-milestone part remains, noted in the last column), `pending` (milestone not reached).

| ID | Spec item | Status | Remaining / notes |
| --- | --- | --- | --- |
| S1-01 | Raw file hashes in data/raw/MANIFEST match the downloaded files | implemented | `real_data` |
| S1-02 | Every item_id unique; no item in two splits | implemented | |
| S1-03 | No pool item has Jaccard >= 0.9 with any probe/test item | implemented | exact check on real data is `real_data` |
| S1-04 | FinQA executor reproduces exe_ans on >= 99% of train; failures logged | implemented | `real_data` |
| S1-05 | TAT-QA evaluator verifies >= 97% arithmetic and >= 99% span items | implemented | `real_data` |
| S1-06 | answers_match passes a hand-written suite of 100 cases | implemented | |
| S1-07 | Each template passes 50 hand-checked items; LLM checks >= 90% human validation | partial | human labels via `annotate export q3_templates` / `q3_validation`; truth-condition property tests automated |
| S1-08 | Every pool item and negative has six Jev rows, probabilities sum to 1 +- 1e-6 | implemented | real Jev client pending TypeSafe docs |
| S1-09 | Probabilities in [0, 1]; no pool item has null R or I | implemented | |
| S1-10 | Rebuild from raw reproduces pool.parquet byte for byte; Jev within tolerance | implemented | |
| S1-11 | qa_tokens <= 2,048 for pool/probe/fixed_demo, <= 4,096 for test | implemented | |
| S1-12 | Probe, fixed-demo and test sets reach target sizes after exclusions | implemented | |
| S1-13 | stable_seed equal across processes; no module calls built-in hash | implemented | |
| S1-14 | build_messages snapshot tests, 10 items per format | implemented | |
| S1-15 | No #neg answer matches its gold, incl. zero-valued and boolean cases | implemented | |
| S1-16 | MockJev and the real client pass the same interface test: argument types, one P(yes) in [0, 1] per question ID, same handling of a 429 with retry-after | partial | real client tested against a fake HTTP endpoint with test-only response fields; the real field names must come from TypeSafe's API reference |
| S1-17 | Every scored item has exactly one batched six-question request in jev_requests.parquet (three if repeats), answers mapped to the right question IDs | implemented | on MockJev; `api` run pending |
| S1-18 | Batched-versus-single-question check on 200 items recorded; full run used the selected mode | partial | logic and both outcomes tested on mocks; the real 200-item check is `api` |
| S2-01 | Every item has one row; S non-null for pool items | pending | M4 |
| S2-02 | Answer-span assertion passes or boundary adjusted and logged; < 1% adjusted | pending | M4 |
| S2-03 | vLLM and fp32 Transformers NLL within 0.05 nats on 200 items | pending | M4, `gpu` |
| S2-04 | Unparsed <= 5% overall and <= 10% per dataset | pending | M4 |
| S2-05 | 50-item hand check of extract_answer shows no errors | pending | M4, human |
| S2-06 | Same seeds reproduce every acc | pending | M4 |
| S2-07 | S uniform on [0, 1] within each dataset (KS p > 0.05) | pending | M4 |
| S2-08 | answer_nll unit tests: toy 3-token sum exact; 20 hand-picked spans exact | pending | M4 |
| S2-09 | Chat template renders no thinking/reasoning block | pending | M4 |
| S2-10 | Sampling 20 items in a fresh process reproduces identical text | pending | M4 |
| S3-01 | Comparators non-null for every pool item; judge_quality >= 99% | pending | M5 |
| S3-02 | Self-consistency covers exactly the stored 3,000-item subsample | pending | M5 |
| S3-03 | Synthetic AUROC 0.8 recovered within 0.02; cluster bootstrap keeps pairs | pending | M5 |
| S3-04 | Every A1–A6 table follows the shared schema with non-null n | pending | M5 |
| S3-05 | gates.json validates against conf/ schema; medians match recomputation | partial | schema validation implemented (M1); medians at M5 |
| S3-06 | Changing r_threshold in D4 recomputes medians and cell counts | pending | M5 |
| S3-07 | icl stage and Specs 5–6 exit at start-up when derived.halt is true | partial | guard in place for the placeholder stages; re-tested when each stage is built |
| S3-08 | judge_quality for every #neg; A1 uses exactly the 1,000 paired originals | pending | M5 |
| S3-09 | Analysis frame: one row per pool original; cells/quintiles match; R_eff in [0, 1] | pending | M5 |
| S3-10 | Synthetic D1-fired gates.json: arith R_eff = 1, lookup R_eff = R | pending | M5 |
| S3-11 | update_gates takes a file lock; two writers lose no update | implemented | |
| S3-12 | H1 row in hypotheses.csv matches the rule on A1 and A2 | pending | M5 |
| S4-01 | Projection fidelity within 0.02; SGD sign check passes | pending | M6 |
| S4-02 | Recomputing G for 50 items in a fresh process identical to 1e-5 | pending | M6 |
| S4-03 | LoRA-wrapped NLL equals unwrapped HF NLL within 1e-4 before any backward pass | pending | M6 |
| S4-04 | Same condition and draw give identical demo_ids and predictions | pending | M6 |
| S4-05 | Demonstrations only from the test item's dataset (FinQA for FinanceBench) | pending | M6 |
| S4-06 | No demonstration is a test, probe, fixed_demo or derived item | pending | M6 |
| S4-07 | zero_shot pilot <= 5% unparsed and nonzero accuracy per dataset | pending | M6 |
| S4-08 | Pilot timing recorded | pending | M6, `gpu` |
| S4-09 | compute_influence runs without gates.json; run_condition exits when halted | pending | M6 |
| S4-10 | Per-example gradients: alone equals within a batch of 8 at batch size 1 | pending | M6 |
| S4-11 | Every prompt passes the length assertion; < 20 eligible raises | pending | M6 |
| S4-12 | Tiny profile: n_draws files per condition, one row per test item | pending | M6 |
| S5-01 | Condition files exist; counts match; no probe/test/#neg/Spec 6 item | pending | M7 |
| S5-02 | Cell/quintile assignment reads gates.json medians | pending | M7 |
| S5-03 | Planted interaction recovered (sign flip) in >= 90% of 100 simulations | pending | M7 |
| S5-04 | Planted condition effect recovered; Holm over exactly 6 contrasts | pending | M7 |
| S5-05 | fe_logit matches statsmodels Logit with dummies within 1e-4 | pending | M7 |
| S5-06 | Bootstrap replicates seeded and reproducible | pending | M7 |
| S5-07 | Figures regenerate from results/ alone; D5/D6 validate against schema | pending | M7 |
| S5-08 | Null simulations: <= 7% false positives (E1-G H4, H2 C-model) | pending | M7 |
| S5-09 | Condition files regenerate from the frame; one label edit changes exactly the affected files | pending | M7 |
| S5-10 | hypotheses.csv: one row per hypothesis x proxy; rerun does not duplicate | partial | record_verdict upsert implemented (M1); full-run check at M7 |
| S5-11 | D5 uses only the pool-mean contrast and H4 rule | pending | M7 |
| S6-01 | e3 exits on halt/halt_spec6; runs with override reason | partial | guard in place for the placeholder stage; re-tested at M8 |
| S6-02 | Corrupted answers fail answers_match, keep precision/units, absent from context | pending | M8 |
| S6-03 | Premise-deleted items differ by exactly one non-support sentence and still verify | pending | M8 |
| S6-04 | Corrupted and premise-deleted sets share identical base items | pending | M8 |
| S6-05 | Base originals have R_eff >= r_threshold; disjoint from low-R (incl. cells_r_filter = false) | pending | M8 |
| S6-06 | Draw j of e3_corrupt/e3_premdel uses exactly e3_orig's base items, same order | pending | M8 |
| S6-07 | Bootstrap replicates carry the same multiset of draw indices across paired conditions | pending | M8 |
| S6-08 | Mismatched LoRA/projection seeds raise | pending | M8 |
| S6-09 | Variant Jev scores share battery and model version; cached originals not re-sent | pending | M8 |
| S6-10 | Planted paired dG detected >= 90% of 200 runs; null <= 7% | pending | M8 |
| S6-11 | Failed manipulation check gives manipulation_failed | pending | M8 |
| S6-12 | No e3_* item in Spec 5 files; Spec 5 results unchanged after Spec 6 | pending | M8 |
