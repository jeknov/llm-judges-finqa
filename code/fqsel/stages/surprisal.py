"""Stage `surprisal` (Spec 2): teacher-forced answer NLL and sampled accuracy of model.base.

One row per non-derived item in every split, including FinanceBench; `#neg` items are not scored.
S is the within-dataset percentile of nll_mean, (rank - 0.5) / n with average ranks, for pool items only.
acc is the share of n_samples `reason`-format samples whose extracted answer matches the gold; all samples
of an item use stable_seed(global_seed, item_id, "surprisal"). Raw samples go to
scores/surprisal_samples.parquet (Spec 2 names runs/surprisal/samples.parquet; runs/ holds the
DVC-tracked result snapshots, so the samples sit next to the scores instead).
"""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from fqsel.backends import make_backend
from fqsel.backends.base import LMBackend
from fqsel.ids import stable_seed
from fqsel.io import read_table, write_table
from fqsel.match import answers_match, modal_group
from fqsel.nll import answer_nll
from fqsel.perturb import gold_for_match
from fqsel.prompts import build_messages, extract_answer


def within_dataset_percentile(nll_mean: pd.Series, dataset: pd.Series) -> pd.Series:
    """(rank - 0.5) / n of nll_mean within each dataset, average ranks for ties; in (0, 1)."""
    ranks = nll_mean.groupby(dataset).rank(method="average")
    sizes = dataset.map(dataset.value_counts())
    return (ranks - 0.5) / sizes


def _batches(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i: i + size]


def run(cfg: DictConfig, backend: LMBackend | None = None) -> None:
    sc = cfg.surprisal
    backend = backend or make_backend(cfg)
    pool = read_table("pool", cfg)
    items = pool[~pool["is_derived"]].sort_values("item_id").reset_index(drop=True)
    records = items.to_dict("records")
    size = int(sc.batch_size)

    nll = pd.concat([answer_nll(backend, [build_messages(it, "direct", with_answer=True) for it in chunk])
                     for chunk in _batches(records, size)], ignore_index=True)

    samples = []
    for chunk in _batches(records, size):
        prompts = [backend.chat_text(build_messages(it, "reason"), add_generation_prompt=True) for it in chunk]
        seeds = [stable_seed(cfg.global_seed, it["item_id"], sc.seed_label) for it in chunk]
        texts = backend.generate(prompts, int(sc.n_samples), float(sc.temperature), float(sc.top_p),
                                 int(sc.max_new_tokens), seeds)
        for it, seed, outs in zip(chunk, seeds, texts, strict=True):
            gold = gold_for_match(it)
            for k, text in enumerate(outs):
                answer = extract_answer(text, it["answer_type"])
                samples.append({"item_id": it["item_id"], "k": k, "seed": int(seed % 2**31), "text": text,
                                "answer": answer,
                                "correct": bool(answer is not None and answers_match(answer, gold, it["answer_type"]))})
    samples = pd.DataFrame(samples)

    rows = []
    by_item = samples.groupby("item_id", sort=False)
    for it, (_, n_row) in zip(records, nll.iterrows(), strict=True):
        g = by_item.get_group(it["item_id"]).sort_values("k")
        modal, msize = modal_group(list(g["answer"]), it["answer_type"])
        rows.append({"item_id": it["item_id"], "nll_sum": n_row["nll_sum"],
                     "n_answer_tokens": int(n_row["n_answer_tokens"]), "nll_mean": n_row["nll_mean"],
                     "acc": float(g["correct"].mean()), "n_correct": int(g["correct"].sum()),
                     "n_unparsed": int(g["answer"].isna().sum()), "modal_answer": modal,
                     "modal_share": msize / len(g), "boundary_adjusted": bool(n_row["boundary_adjusted"])})
    out = pd.DataFrame(rows)
    in_pool = (items["split"] == "pool").to_numpy()
    out["S"] = np.nan
    out.loc[in_pool, "S"] = within_dataset_percentile(out.loc[in_pool, "nll_mean"],
                                                      items.loc[in_pool, "dataset"]).to_numpy()

    unparsed = samples["answer"].isna().groupby(samples["item_id"].map(items.set_index("item_id")["dataset"])).mean()
    meta = {"backend": cfg.backend, "model": cfg.model.base.name, "revision": cfg.model.base.get("revision"),
            "chat_template_sha256": hashlib.sha256(str(backend.tokenizer.chat_template).encode()).hexdigest(),
            "vllm_version": _vllm_version(cfg), "unparsed_share": float(samples["answer"].isna().mean()),
            "unparsed_share_by_dataset": {k: float(v) for k, v in unparsed.items()},
            "boundary_adjusted_share": float(out["boundary_adjusted"].mean())}
    write_table("surprisal", out[["item_id", "nll_sum", "n_answer_tokens", "nll_mean", "S", "acc", "n_correct",
                                  "n_unparsed", "modal_answer", "modal_share", "boundary_adjusted"]],
                cfg, extra_meta={"surprisal": meta})
    write_table("surprisal_samples", samples, cfg)
    print(f"surprisal: {len(out)} items; mean acc {out['acc'].mean():.3f}; unparsed {meta['unparsed_share']:.1%}; "
          f"boundary adjusted {meta['boundary_adjusted_share']:.2%}")
    if cfg.profile != "tiny":   # Spec 2: a formatting failure must not masquerade as difficulty
        bad = [f"overall {meta['unparsed_share']:.1%} > {sc.unparsed_max:.0%}"] if meta["unparsed_share"] > sc.unparsed_max else []
        bad += [f"{k} {v:.1%} > {sc.unparsed_max_per_dataset:.0%}" for k, v in meta["unparsed_share_by_dataset"].items()
                if v > sc.unparsed_max_per_dataset]
        if meta["boundary_adjusted_share"] > sc.boundary_adjusted_max:
            bad.append(f"boundary adjusted {meta['boundary_adjusted_share']:.2%} > {sc.boundary_adjusted_max:.0%}")
        if bad:
            raise RuntimeError("surprisal acceptance limits exceeded (outputs written for inspection): " + "; ".join(bad))


def _vllm_version(cfg: DictConfig) -> str | None:
    if cfg.backend != "vllm":
        return None
    import vllm

    return vllm.__version__
