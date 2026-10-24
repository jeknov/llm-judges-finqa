# Implication-Weighted Data Selection for Financial QA — Protocol v1 (Training-Free)

Oct 23, 2026 · @Jekaterina

## Relation to the specs

The six implementation specs refine this protocol in a few documented places (validation negatives, the surprisal percentile, a fixed k = 4, per-dataset cell counts in D4, and others). The full list, with reasons, is the "Deviations from protocol v1" table in [Spec 1](https://claude.ai/artifact/5JKwCAyWXR5ouyNk6oNQiz). The text below is protocol v1 as agreed; only the compute estimate has been corrected.

## Thesis

The value of a training example for adapting a model to a task is driven by how strongly it implies task-relevant consequences, weighted by how much it would change the model and gated by whether it is true. Certainty in the ordinary sense, meaning confidence that the datum is correct, is the gate rather than the driver.

The motivating case is diesel price prediction during a war. An institutional forecast reports strong production, while a drone operator reports that the main production site was destroyed half an hour ago. The drone report is more valuable, yet it is the less reliable claim. It wins because, if true, it entails a supply shock in one or two inferential steps, it contradicts the prior, and it supersedes older evidence.

As such, the study decomposes value into three quantities: reliability, P(datum is true); implication strength, P(task-relevant consequence | datum is true); and surprisal, how far the datum would move the base model. The central claim is a moderation: implication strength determines whether surprising data is useful. A result that reduces to "clean labels help" or "easy data helps" is a null result.

In this variant, "value" means predicted value for adaptation, measured by two proxies that require no training. The claim is therefore about which data a model is positioned to learn from, not about the outcome of a completed fine-tuning run.

## Constructs and operationalization

Reliability and implication strength come from Jev; surprisal comes from the base model. Jev returns a typed decision with a probability per label, so each battery question yields a probability and an entropy at negligible cost.

| Construct | Source | Measure | Item-level score |
| --- | --- | --- | --- |
| Reliability R | Jev, Q1 | P(supported) | Probability of "yes" |
| Implication strength I | Jev, Q2 and Q3 | P(determined), P(consequence holds) | Mean of the two "yes" probabilities |
| Surprisal S | Base model | Mean token NLL of the gold answer; base-model accuracy over 5 samples | NLL, with accuracy as a robustness check |
| Jev indecision | Jev, all questions | Label entropy | Max entropy across Q1–Q3 |

**Jev battery** (v0 wording, frozen before any value measurement):

1. Q1, support: "Given the context and question, is the stated answer supported by the context?" Labels: yes, no.
2. Q2, determinacy: "Do the context and question determine this answer uniquely, such that no materially different answer would also be defensible?" Labels: yes, no.
3. Q3, consequence: "Assume the stated answer is true. Does it entail the following check?" The check is templated per item: the direction or sign of a change, consistency with a stated total, or the ordering of two quantities. Labels: yes, no.

Q3 checks are generated deterministically from FinQA and TAT-QA programs where possible, and by an LLM with Outlines schema constraints otherwise; generated checks are spot-validated by human annotators on CSV files, which are returned with their labels. Composite value is V = R × I, with S entering as a crossed factor so that its interaction can be estimated. A paraphrased battery (Q1′–Q3′) is scored on the full pool to test robustness.

### Value proxies

**Proxy G, first-order gradient influence.** For each pool item z, compute the gradient of the answer loss with respect to a fixed parameter subset of the frozen base model: LoRA matrices with random A and zero-initialized B, so that the gradient with respect to B is nonzero, projected to a fixed dimension with a random projection. Influence is the cosine similarity between the item gradient and the mean gradient over a probe set P:

```latex
G(z) = \cos\!\left(\nabla_\theta \ell(z),\; \frac{1}{|P|}\sum_{p \in P} \nabla_\theta \ell(p)\right)
```

G(z) approximates how much a single update on z would reduce probe loss. It is the proxy closest to the fine-tuning mechanism, and it yields an item-level score for the whole pool.

**Proxy C, in-context demonstration gain.** For a selection condition, draw k demonstrations from its eligible items and measure base-model accuracy on test items, averaged over many independent draws. The gain is accuracy relative to the random-demonstration condition. It is the proxy closest to observable accuracy, and it operates on subsets rather than items.

The two proxies have different failure modes. G ignores optimizer dynamics beyond the first step; C may be insensitive to label correctness in ways that training is not. Agreement between them is therefore a pre-registered requirement (H6).

## Hypotheses

Six hypotheses are pre-registered. H4 carries the study; H1 and H6 are validity gates, and H2, H3 and H5 support the mechanism claim.

| ID | Hypothesis | Test | Supports thesis if |
| --- | --- | --- | --- |
| H1 | Jev scores are valid and capture something existing scores do not | E0 | AUROC of R vs verified correctness ≥ 0.70; Spearman ρ(I, base-model accuracy) < 0.5; ρ(I, prompted PVI) < 0.8 and ρ(I, ICL-reducible loss) < 0.8 |
| H2 | High-I data has higher predicted value than random after difficulty control | E1 | Residualized-I beats random under both G and C, and beats the AlpaGasus-style baseline |
| H3 | Predicted value rises monotonically with I | E2 | Positive trend across I quintiles under both proxies |
| H4 | Implication strength moderates surprisal: S lowers predicted value at low I and raises it at high I | E1-G, E1-C | Sign-flipping I × S interaction in the item-level G regression; matching cell ordering under C |
| H5 | High-I, high-S items with false labels have negative value | E3 | Corrupted items have lower G than their high-R originals, with G below the pool median; corrupted demonstrations reduce C relative to random |
| H6 | The two proxies agree | All | Spearman ρ ≥ 0.5 between condition rankings under G and C |

H4 tests usefulness only. The harm reported by Gekhman et al. (2024), where unknown examples are fitted slowly and then increase hallucination, is a training-dynamics effect that neither proxy reproduces. Without the optional anchor, the study makes no claim about training-induced hallucination.

H5 is the propaganda drone report in QA form. Under proxy C it carries a specific risk: prior work on classification tasks (Min et al. 2022, to be verified) found that demonstration label correctness matters less than expected. A null H5 under C with a positive H5 under G is therefore interpretable rather than contradictory, and is reported as such.

## Setup

One frozen 7–8B open instruct model is the subject of every measurement; nothing is fine-tuned except the optional anchor.

| Component | Choice |
| --- | --- |
| Pool | FinQA train plus TAT-QA train, deduplicated |
| Probe set P | 500 items from the FinQA and TAT-QA dev splits, stratified by dataset and question type; disjoint from test |
| Model | One 7–8B open instruct model, not finance-tuned, frozen throughout |
| Gradients | LoRA rank 16 on attention projections, random A, zero B; random projection to 8,192 dimensions |
| Demonstrations | k = 4 (reduced to k = 2 where contexts exceed the length budget); 50 independent draws per condition |
| In-distribution test | 500 items each from FinQA test and TAT-QA test |
| Out-of-distribution test | FinanceBench subset with verifiable numeric answers |
| Harness | vLLM with prefix caching for C; Inspect AI for scoring; Hydra for the condition grid; DVC for subset manifests |

The metrics under C are exact-match accuracy with numeric tolerance and the CBI rate, defined as the share of test items answered consistently across 5 samples but incorrectly, using the Thread 2 definition. Verified correctness for E0 and E3 comes from executing the gold FinQA programs and TAT-QA derivations, not from Jev or an LLM judge.

## Experiments

All experiments are inference or single backward passes on the frozen model. The optional anchor is the only training.

**E0, measurement validity.** Score the full pool with both batteries; report the AUROC of R against verified correctness, split into lookup and arithmetic items; report the Spearman correlation of I and R with base-model accuracy, test-retest agreement between batteries, and agreement with 5-sample self-consistency of a strong LLM; hand-audit 150 items stratified by R × I cells. The PVI comparator is approximated by the base model's answer log-likelihood with and without the context. The RHO-LOSS comparator is approximated by the reduction in answer loss when 4 held-out demonstrations are prepended.

**E1-G, item-level influence regression.** Compute G(z) for every pool item. Regress G on I, S, I × S, R and base-model accuracy. This is the primary H4 test and requires no subset construction.

**E1-C, matched-condition demonstration study.**

| Condition | Eligible items | Draws |
| --- | --- | --- |
| Random | Whole pool | 50 |
| Top-R | Top 20% by reliability (noise-filtering baseline) | 50 |
| Top-V | Top 20% by R × I | 50 |
| Top-I residualized | Top 20% by I after regressing out base-model accuracy, within R ≥ 0.9 | 50 |
| LLM-judge quality | Top 20% by AlpaGasus-style quality score | 50 |
| IFD | Top 20% by instruction-following difficulty | 50 |
| I × S cells | Four cells: high/low I crossed with high/low S, within R ≥ 0.9 (median splits) | 4 × 50 |

**E2, dose-response.** Five I quintiles within R ≥ 0.9: mean G per quintile from E1-G, and 50 demonstration draws per quintile under C.

**E3, reliability gate and causal manipulation.** Three variants of the high-I, high-S cell, evaluated under both proxies. Low-R uses naturally occurring low-reliability items in that cell. Corrupted uses high-R items with the numeric answer perturbed so that it still looks determinate. Premise-deleted keeps the gold answer valid but deletes a non-essential context sentence, which lowers determinacy without touching correctness. Under G, each variant is compared with its original item in a paired design, which gives a within-item estimate unavailable in v0.

**Optional anchor.** Fine-tune the four I × S cells with LoRA at one seed and 1,000 items each, for 4 runs. This tests whether the proxy ordering of the cells matches the fine-tuned ordering. It is the only route to the training-harm half of H4.

## Decision points and kill conditions

Each gate is checked before the next stage spends compute. A failed gate changes the claim rather than being explained away.

| Gate | When | Condition | Action if triggered |
| --- | --- | --- | --- |
| D1, validity | After E0 | AUROC of R vs verified correctness < 0.70 on arithmetic items | Restrict Q1 to lookup items, or replace it with program execution where gold programs exist |
| D2, difficulty proxy | After E0 | ρ(I, base-model accuracy) ≥ 0.7 | I is a difficulty proxy; reframe or kill |
| D3, separability | After E0 | R and I correlate at ρ ≥ 0.8 | Revise Q2 and Q3 before E1 |
| D4, cell sizes | After E0 | Any I × S cell with R ≥ 0.9 has fewer than 200 items | Relax the reliability threshold to R ≥ 0.8 for all cells |
| D5, core claim | After E1-G | No positive association between residualized I and G, and no sign flip in I × S | The thesis collapses into noise filtering; report E0 as a measurement paper |
| D6, proxy agreement | After E1-C | Condition rankings under G and C correlate at ρ < 0.3 | Report both; make no claim that rests on one proxy alone; run the anchor if any compute is available |

D5 retains a fallback: if the moderation claim fails, E0 alone is a publishable construct-validity study of whether a calibrated judge separates reliability from implication on verifiable financial QA.

## Statistical analysis

E1-G uses a linear model on item-level influence with dataset as a fixed effect:

```latex
G_i = \beta_0 + \beta_1 I_i + \beta_2 S_i + \beta_3 (I_i \times S_i) + \beta_4 R_i + \beta_5\,\text{acc}_i + \gamma_{\text{dataset}} + \varepsilon_i
```

H4 predicts β3 > 0 together with a negative simple slope of S at low I. Confidence intervals come from a bootstrap that resamples probe items, so that uncertainty from the probe set is included.

E1-C uses a mixed-effects logistic regression on test-item correctness, `correct ~ condition + (1 | test_item) + (1 | draw)`, with random as the reference level, fit in lme4 through rpy2. E3 under G uses paired differences with a Wilcoxon signed-rank test. H6 uses Spearman correlation between condition rankings with a bootstrap over draws.

Contrasts against random are Holm-corrected within experiment. In-distribution and out-of-distribution results are reported separately.

## Compute and timeline

Proxy G needs one forward and backward pass per pool item and per probe item on the frozen model, which is a few GPU-hours. Proxy C is inference only: about 14 conditions × 50 draws × 1,000 test items, roughly 700,000 prompts. With prefix caching, the shared demonstrations are processed once per draw, so this takes under a day on one 8-GPU node, or a few days on a single GPU; 5-sample consistency checks on every item are the term that pushes it toward a full day. Jev scoring costs a few dollars at $0.044 per 1,000 judgments. All figures are estimates to confirm on a one-condition pilot.

1. Week 1: build the pool, generate Q3 checks, score the batteries, compute surprisal and the prompted comparators, run E0 and gates D1–D4.
2. Week 2: compute G for the pool and probes, run E1-G, E2 under G and E3 under G, then gate D5.
3. Week 3: run E1-C, E2 and E3 under C, gate D6, analysis and the 4-page draft. The anchor runs in parallel if compute allows.

The headline figure is the item-level I × S interaction plot from E1-G, with the E1-C cell means overlaid.

## Related work and novelty risks

The closest ancestor is RHO-LOSS. The novel claim is that judge-measured implication strength moderates whether surprising data is useful for adaptation, which reconciles conflicting findings on easy versus hard training data.

| Work | What it established | How this study differs |
| --- | --- | --- |
| [RHO-LOSS](https://arxiv.org/abs/2206.07137) (Mindermann et al., ICML 2022) | Select points that are learnable, worth learning, and not yet learnt | Learnability there is holdout-model loss; here reliability and implication are measured externally. ICL approximation is an E0 comparator |
| [V-usable information / PVI](https://arxiv.org/abs/2110.08420) (Ethayarajh et al.) | Per-instance usable information the input carries about the label | Closest existing measure of implication strength; prompted approximation compared under H1 |
| [Dataset cartography](https://aclanthology.org/2020.emnlp-main.746) (Swayamdipta et al., EMNLP 2020) | Hard-to-learn items are often mislabeled; ambiguous items drive generalization | Partly contrary to the thesis; requires training dynamics, so only discussed here |
| [Fine-tuning on new knowledge](https://aclanthology.org/2024.emnlp-main.444) (Gekhman et al., EMNLP 2024) | Unknown examples are learned slowly and then increase hallucination | Motivates H4; the harm side is out of scope without the anchor |
| [KaFT](https://arxiv.org/abs/2505.15480) | Large knowledge conflicts in domain QA data cause sharp drops | I is proposed as what separates useful from harmful conflict |
| LESS (Xia et al.), TracIn (Pruthi et al.) | Gradient-similarity influence for data selection | Proxy G follows this approach without a warmup run; citations to verify |
| [Demonstrations as tuning data](https://aclanthology.org/2026.acl-long.45.pdf) (ACL 2026) | Good in-context demonstrations are effective instruction-tuning examples | Justifies proxy C |
| Min et al. (2022) | Demonstration label correctness matters less than expected in classification | Main risk for H5 under C; to verify |
| AlpaGasus, DEITA, [QuRating](https://arxiv.org/abs/2402.09739) | LLM-judge quality scores | No split between truth and implication; AlpaGasus-style score is an E1-C baseline |
| IFD, [SelectIT](https://arxiv.org/abs/2402.16705), [InfoSFT](https://arxiv.org/abs/2605.14967) | Model-internal difficulty and uncertainty; conflicting findings on easy versus hard data | Surprisal is one axis; its effect is predicted to flip sign with I |
| [JEV-as-a-Judge](https://arxiv.org/abs/2609.26550) (Li et al.) | Jev confidence as an escalation signal for judging outputs | No prior use of Jev to value training data |

Three risks need explicit handling. First, if I correlates with prompted PVI or the ICL-reducible loss above 0.8, the construct is not new. Second, a training-free study invites the objection that proxies are not training; H6 and the optional anchor are the responses. Third, Jev is weakest on derivation checking and persuasive wrong answers, which overlaps FinQA arithmetic and the corrupted E3 variant; D1 exists for this reason.

## Open decisions

- [ ] Which base checkpoint: untuned on finance, with a known tokenizer, ideally reusable for Thread 1.
- [ ] Which parameter subset for G: LoRA on attention only, or also MLP projections.
- [ ] Whether the 4-run anchor is affordable, which decides whether any training-harm claim is possible.
- [ ] Which workshop venue and deadline.
- [ ] Whether to frame H5 and the CBI link as a Thread 2 bridge or keep the study self-contained.
