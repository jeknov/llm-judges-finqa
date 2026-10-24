# Spec 4 — Value Proxies G and C

Oct 22, 2026 · @Jekaterina

## Purpose

This spec implements the two value proxies as reusable engines; it defines how value is measured, not which items are compared. Spec 5 decides the selection conditions and Spec 6 the manipulated items, and both call these engines. It is Step 4 of protocol v1.

Two entry points are exposed:

- `compute_influence(item_ids) -> influence.parquet rows`, which computes G for any items, including Spec 6 variants.
- `run_condition(name, eligible_ids, n_draws) -> runs/icl/<name>/`, which runs proxy C for one selection condition.

`compute_influence` does not read `gates.json`, because G depends on no gate; it can start as soon as Specs 1 and 2 have produced the pool and probes, which takes it off the critical path. `run_condition` reads `results/gates.json` and exits if `derived.halt` is true, since its conditions depend on gate-derived labels. Inputs: `pool.parquet` from Spec 1, `model.base`, `build_messages` and `answer_nll` from Specs 1–2. Shared conventions are defined in Spec 1.

## Proxy G: gradient influence

G is the cosine similarity between an item's loss gradient and the mean probe gradient, taken over freshly initialized LoRA parameters on the frozen base model, after a random projection. No parameter is ever updated.

**Parameters.** Attach LoRA adapters with PEFT: rank 16, alpha 32, dropout 0, on the q, k, v and o projections of every layer. A is initialized by PEFT's default, B is zero, and the initialization seed is fixed in config. Because B is zero, only the gradient with respect to B is nonzero; the gradient is taken over the B matrices only.

**Loss.** The `direct`-format sequence from Spec 2, with the loss as the mean NLL over the same answer-token span. Prompt tokens are masked with label −100.

**Per-example gradients.** Batch size 1, one backward pass per item. The forward pass runs in bf16 and the B gradients are cast to fp32 before projection. Flatten all B gradients in a fixed layer order into one vector g of dimension D.

**Projection.** Project g to d = 8,192 dimensions with a fixed Rademacher matrix scaled by 1/√d, generated from a seed in config and applied in chunks so that the full D × d matrix is never materialized. Use the same projection for pool items, probes and Spec 6 variants. Store the projected vectors in fp16.

**Influence.** Let ĝ(z) be the projected gradient of item z, and ḡ the mean of the projected probe gradients.

```latex
G(z) = \frac{\hat g(z) \cdot \bar g}{\lVert \hat g(z) \rVert \, \lVert \bar g \rVert}, \qquad \bar g = \frac{1}{|P|} \sum_{p \in P} \hat g(p)
```

Also store the raw dot product `G_dot` and the norm `grad_norm`. The cosine is primary, as in the protocol. The dot product is reported as secondary, because it is the actual first-order change in probe loss and scales with the norm, which itself correlates with surprisal.

Spec 5 needs uncertainty over the probe set, so the 500 projected probe vectors are saved individually (`grads/probe.npy`, 500 × 8,192), not just their mean. Dataset-specific variants `G_finqa` and `G_tatqa` use only that dataset's probes.

**Sanity checks.** Run before the full pass:

- Projection fidelity: on 100 items, the cosine between two items in projected space must match the unprojected cosine within 0.02.
- Sign check: take one SGD step of size 1e-4 on 20 items with the highest and 20 with the lowest G, separately, and confirm that mean probe loss falls more after the high-G step. This is the only weight update in the spec, and it is discarded.

## Proxy C: demonstration gain

Proxy C runs each test item with k demonstrations drawn from a condition's eligible items, in the `direct` format with greedy decoding, and scores the answers. The `direct` format is used throughout, rather than `reason`, so that a demonstration is exactly its question and gold answer. A reasoning trace would otherwise have to be corrupted along with the answer in Spec 6, and would carry information beyond the item's label.

**Demonstration sampling.** For draw j of condition c, sample 4 items without replacement from the condition's eligible FinQA items and, separately, 4 from its eligible TAT-QA items, using `stable_seed(global_seed, c, j)`. A test item receives the demonstrations of its own dataset; FinanceBench items receive the FinQA set. All test items in a draw share the same demonstrations, which is what makes prefix caching effective. A condition whose eligible list has fewer than 20 items in either dataset is an error, not a silent fallback.

**k.** Always 4. The token budget in Spec 1 (pool items ≤ 2,048 tokens, test items ≤ 4,096, context ≥ 16,384) guarantees that every prompt fits, so there is no per-dataset reduction. A runtime assertion checks each prompt's length anyway.

**Prompt assembly.** `build_messages(test_item, "direct", demos)` from Spec 1, with the demonstrations shuffled by `stable_seed(global_seed, c, j, "order")`. No other code assembles prompts.

**Decoding and scoring.** Greedy, `max_new_tokens = 64`, stop at end-of-turn. Extract with `extract_answer` and score with `answers_match`, both from Specs 1–2.

**Zero-shot baseline.** A condition named `zero_shot` with k = 0 and one draw. Gains in Spec 5 are reported against both `zero_shot` and `random`.

**CBI sampling.** For the first 10 draws of each condition only, also sample 5 answers per test item at temperature 0.7 and top-p 0.95, seeded with stable\_seed(global\_seed, c, j, test item ID, "cbi"). An item is consistent if its modal answer has a share of at least 0.8 (4 of 5). The CBI rate is the share of test items that are consistent and whose modal answer is wrong.

## Test sets

Proxy C is scored on three sets: 500 FinQA and 500 TAT-QA test items from Spec 1 (in-distribution), and a FinanceBench subset (out-of-distribution). Results for the three are always reported separately.

**FinanceBench subset.** Built and filtered in Spec 1 (Pool construction), with `dataset = financebench` and `split = test`. Out-of-distribution results are hypothesis-tested only if at least 50 items survive that filter; otherwise they are reported as descriptive.

FinanceBench items receive the FinQA demonstrations of each draw, since FinQA is the closer format.

They are scored with `extract_answer(text, "arith")` and `answers_match`, like every other test item.

**Fixed order.** Test items are processed in a fixed order sorted by `item_id`, and every condition and draw uses the identical list, so per-item comparisons across conditions are paired.

## Outputs and run interface

Each engine writes item-level rows only; aggregation and statistics belong to Spec 5. Runs are idempotent: an existing output with a matching config hash is skipped, which lets interrupted runs resume.

**`scores/influence.parquet`**, one row per scored item.

| Column | Type | Meaning |
| --- | --- | --- |
| `item_id` | str | Spec 1 ID, including Spec 6 suffixes |
| `G` | float | Cosine with the mean of all probe gradients |
| `G_finqa`, `G_tatqa` | float | Cosine with each dataset's probe mean |
| `G_dot` | float | Dot product with the mean of all probe gradients |
| `grad_norm` | float | Norm of the projected gradient |
| `lora_seed`, `proj_seed` | int | Initialization and projection seeds |

Projected gradients are saved as `grads/pool.npy` and `grads/probe.npy`, with an index file mapping rows to `item_id`.

**`runs/icl/<condition>/<draw>.parquet`**, one row per test item.

| Column | Type | Meaning |
| --- | --- | --- |
| `condition`, `draw` | str, int | Run identity |
| `test_item_id` | str | Test item |
| `demo_ids` | list\[str\] | Demonstrations used, in prompt order |
| `prediction` | str | Extracted greedy answer |
| `correct` | bool | `answers_match` result |
| `cbi_modal_answer`, `cbi_modal_share`, `cbi_correct` | str, float, bool | Draws 0–9 only; null otherwise |

**Run interface.** Specs 5 and 6 build conditions as YAML files under `conf/conditions/`. Each file names the condition, lists `eligible_ids` or a selection rule resolvable from stored scores, and sets `n_draws`. A Hydra multirun over these files is the whole C experiment.

A condition file may give a `demo_plan` instead of `eligible_ids`: a mapping from draw index to the 4 demonstration IDs per dataset, in prompt order. `run_condition` then uses the plan verbatim and skips sampling and shuffling. Spec 6 uses this to pair its conditions draw by draw. A file with both fields, or neither, is rejected.

## Acceptance tests and open items

Spec 4 is done when both engines pass these tests on a 200-item pilot, before any full run.

- [ ] Projection fidelity within 0.02, and the SGD sign check passes.
- [ ] Recomputing G for 50 items in a fresh process gives identical values to 1e-5.
- [ ] Before any backward pass, the LoRA-wrapped model's answer NLL equals the unwrapped Hugging Face model's, in the same precision, to within 1e-4 nats, which confirms that B = 0 leaves outputs unchanged. The comparison is not against Spec 2's vLLM values, whose kernels differ.
- [ ] Two runs of the same condition and draw produce identical `demo_ids` and predictions.
- [ ] Every test item in a draw has demonstrations only from its own dataset, or FinQA for FinanceBench items.
- [ ] No demonstration is a test, probe, `fixed_demo` or derived item, unless the condition file belongs to Spec 6 and names derived items explicitly.
- [ ] The `zero_shot` pilot has at most 5% unparsed predictions and nonzero accuracy in each dataset; Spec 2's sampled accuracy uses the reason format and is not comparable.
- [ ] Pilot timing for one condition is recorded and used to confirm the compute estimate before the full run.

* [ ] `compute_influence` runs successfully with no `gates.json` present; `run_condition` exits with a clear error when `derived.halt` is true.
* [ ] Gradients are per example: G computed for an item alone equals G computed for it within a batch of 8 run with batch size 1, and a test fails if gradient accumulation across items is ever enabled.
* [ ] Every assembled prompt passes the length assertion, and a condition with fewer than 20 eligible items in a dataset raises an error.
* [ ] On the tiny profile, each condition's run directory holds exactly `n_draws` files, each with one row per test item.

**Open items.**

- Confirm the FinanceBench release fields (question, answer, evidence text) and how many items survive the filter.
- Decide whether the GPU budget allows CBI sampling on all 10 draws, or whether 5 draws suffice.
