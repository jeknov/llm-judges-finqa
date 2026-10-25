# Spec 2 — Base-Model Surprisal

Oct 22, 2026 · @Jekaterina

## Purpose

This spec measures how surprising each pool item is to the frozen base model, producing surprisal S and base-model accuracy for every pool item. It is Step 2 of protocol v1 and can run in parallel with Jev scoring, since it needs only `pool.parquet` from Spec 1.

S is the teacher-forced loss of the gold answer. Base-model accuracy is the share of sampled answers that match the gold. The two play different roles downstream: S is a crossed factor in the I × S design, and accuracy is the difficulty covariate removed in residualization and checked in gate D2. Shared conventions (item IDs, file layout, config, seeds, `render_item`) are defined in Spec 1.

## Base model setup

One checkpoint, `model.base`, is used for this spec and Specs 3–6 without change. It runs in bf16 on vLLM, with the model's own chat template applied through its tokenizer.

**Two prompt formats.** Both use one user turn built from `render_item(item, "qa")` followed by an instruction, and both are produced only by `build_messages` (Spec 1, Shared helpers), which Specs 3–6 reuse.

| Format | Used for | Instruction appended to the user turn | Assistant turn |
| --- | --- | --- | --- |
| `direct` | Teacher-forced NLL (this spec), Spec 3 comparators, Spec 4 gradients and demonstration runs | "Give only the final answer, with units if any." | `Answer: {gold_answer}` |
| `reason` | Sampled accuracy (this spec) | "Reason briefly, then give the final answer on its own last line as: Answer: \<value>" | generated |

**Fixed settings.**

- Context length: the checkpoint must support at least 16,384 tokens; the token budget in Spec 1 then guarantees that every prompt in Specs 2–6 fits, so no spec truncates or shrinks k.
- Thinking or reasoning modes the chat template exposes are switched off, so that every generation follows the `reason` instruction alone.
- The tokenizer, chat template and vLLM version are recorded in each output file's metadata.

## Teacher-forced answer NLL

S is the mean negative log-likelihood per token of the gold answer, scored in the `direct` format with no sampling. Only the answer tokens count; the prompt and the ` Answer:  ` prefix do not.

**Procedure.**

1. Build the full sequence: chat-templated user turn, then the assistant turn `Answer: {gold_answer}`, then the template's end-of-turn token.
2. Tokenize once. Locate the answer span by tokenizing the prefix up to and including ` Answer:  ` separately and taking the token positions after it, up to but excluding the end-of-turn token. Assert that the prefix tokens are an exact prefix of the full sequence; if a tokenizer merges across the boundary, move the boundary one token earlier and log the item.
3. Run the sequence through vLLM with `prompt_logprobs` enabled and `max_tokens = 1`, and read the log-probability of each actual token in the answer span.
4. Store `nll_sum` (summed over the span), `n_answer_tokens`, and `nll_mean = nll_sum / n_answer_tokens`.

**Implementation note (25 October 2026).** The boundary is placed after `Answer:`, before the following space. BPE tokenizers, including OLMo 3's, attach a leading space to the next token (` 12`). With the boundary after the space, 11 of 20 hand-picked answers needed boundary adjustment, against the < 1% acceptance limit. With it before the space, none did. The answer span therefore includes the space-prefixed first answer token: decoding it gives the answer with one leading space, the same for every item.

This procedure is implemented once, as `answer_nll(items, fmt, demos)` in `fqsel/nll.py`. Spec 3 calls it with context removed, with fixed demonstrations, and with the instruction removed; Spec 6 calls it on manipulated items. Callers change only the messages, never the span logic.

**Normalization.** Raw NLL depends on answer length and number formatting, which differ between FinQA and TAT-QA. The analysis variable is therefore the within-dataset percentile rank of `nll_mean`, stored as `S` in \[0, 1\]. Median splits for the I × S cells (Spec 5) are taken on this within-dataset rank, so that each cell holds both datasets in proportion.

**Cross-check.** For 200 random items, recompute `nll_sum` with Hugging Face Transformers in fp32 from the same tokens. The two must agree to within 0.05 nats per item; larger gaps indicate a span or template bug.

## Sampled accuracy

Base-model accuracy is the share of 5 sampled `reason`-format answers that match the gold under `answers_match` from Spec 1. It is computed for every pool item; probe and test items are also scored, for descriptive statistics only.

**Sampling parameters.**

| Parameter | Value |
| --- | --- |
| n | 5 |
| temperature | 0.7 |
| top\_p | 0.95 |
| max\_new\_tokens | 512 |
| seed | `stable_seed(global_seed, item_id, "surprisal")`, recorded per item |

**Answer extraction `extract_answer(text)`.** Shared with Specs 4 and 6.

1. Take the last line that matches `^\s*Answer:\s*(.+)$`, case-insensitive.
2. If none matches, take the last number in the final 200 characters for `arith` items, or mark the sample `unparsed` for `lookup` items.
3. Strip trailing punctuation, then pass the string to `answers_match`.

The function's signature is `extract_answer(text, answer_type)`, since step 2 depends on the answer type.

Unparsed samples count as incorrect. If more than 5% of all samples are unparsed, fix the instruction or the extractor before continuing, since a formatting failure would masquerade as difficulty.

**Stored per item.** `acc` (the mean over the 5 samples), `n_correct`, `n_unparsed`, `modal_answer` and `modal_share` (the share of samples giving the most frequent normalized answer). Spec 4 computes its own zero-shot baseline in the direct format, so nothing here feeds proxy C.

## Derived scores and output

Two variables leave this spec for analysis: `S`, the within-dataset percentile of answer NLL, and `acc`, the sampled accuracy. Their Spearman correlation is reported in Spec 3; a value above 0.8 means surprisal adds little beyond difficulty, which Spec 3 records as a finding rather than a gate.

**Output `surprisal.parquet`.** One row per non-derived item in all splits, including FinanceBench; \`#neg\` items are not scored here.

| Column | Type | Meaning |
| --- | --- | --- |
| `item_id` | str | Spec 1 ID |
| `nll_sum` | float | Summed NLL of answer tokens, nats |
| `n_answer_tokens` | int | Tokens in the answer span |
| `nll_mean` | float | `nll_sum / n_answer_tokens` |
| `S` | float | Within-dataset percentile of `nll_mean`, in \[0, 1\]; pool items only |
| `acc` | float | Share of 5 samples correct |
| `n_correct` | int | Correct samples |
| `n_unparsed` | int | Samples without an extractable answer |
| `modal_answer` | str | Most frequent normalized answer |
| `modal_share` | float | Share of samples giving it |
| `boundary_adjusted` | bool | True if the answer span boundary was moved |

The raw generations go to `scores/surprisal_samples.parquet` (item, sample index, seed, text, extracted answer, correct) so that any score can be audited. They were first specified as `runs/surprisal/samples.parquet`; `runs/` now holds the DVC-tracked result snapshots.

## Acceptance tests and open items

Spec 2 is done when these pass and the checkpoint decision is recorded.

- [ ] Every item in all splits has one row; `S` is non-null for every pool item.
- [ ] The answer-span assertion passes, or the boundary was adjusted and logged, for 100% of items; fewer than 1% need adjustment.
- [ ] vLLM and fp32 Transformers NLL agree within 0.05 nats on all 200 cross-check items.
- [ ] Unparsed samples are at most 5% overall and at most 10% in each dataset.
- [ ] A 50-item hand check of `extract_answer` against the raw text shows no extraction errors.
- [ ] Re-running with the same seeds reproduces every `acc` exactly.
- [ ] Within each dataset, `S` is uniform on \[0, 1\] (Kolmogorov–Smirnov p > 0.05), as a percentile should be.

* [ ] `answer_nll` unit tests: on a 3-token toy answer with known log-probabilities the sum is exact, and on 20 hand-picked answers with leading spaces, percent signs and negative numbers the span covers exactly the answer tokens.
* [ ] A test renders one item through the chat template and asserts that no thinking or reasoning block appears.
* [ ] Rerunning sampling for 20 items in a fresh process reproduces identical text, confirming `stable_seed` is used.

**Open items.**

- Choose `model.base`: an open instruct model of 7–8B parameters, not finance-tuned, ideally with the tokenizer used in Thread 1. Record the exact Hugging Face revision hash.
- Confirm that the chosen chat template allows reasoning modes to be switched off.
