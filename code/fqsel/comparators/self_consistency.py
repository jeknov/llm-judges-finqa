"""Self-consistency of the strong LLM (Spec 3, comparators `sc_acc`, `sc_modal_share`).

Subsample: 1,500 pool originals per dataset, stratified by answer_type x R quintile (quintiles within the
dataset), so that agreement with R can be estimated across its range; IDs stored in
data/sc_subsample.parquet. Each item is answered n_samples times in the `reason` format with the P5
sampling settings; sample k uses stable_seed(global_seed, item_id, "sc", k), so a later run with more
samples reuses the cached first ones.

sc_acc: share of samples whose extracted answer matches the gold (answers_match). sc_modal_share: share of
samples in the largest group of mutually matching answers (unextractable answers form no group); the
modal answer is that group's first member, and modal_correct records whether it matches the gold (A4).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from omegaconf import DictConfig

from fqsel.clients.strong_llm import StrongClient, make_strong, sc_sampling, sc_seed
from fqsel.data.splits import sample_stratified
from fqsel.io import read_table, write_table
from fqsel.match import answers_match, modal_group
from fqsel.perturb import gold_for_match
from fqsel.prompts import build_messages, extract_answer

PER_DATASET = 1500


def select_subsample(cfg: DictConfig) -> pd.DataFrame:
    pool = read_table("pool", cfg)
    scores = read_table("jev_scores", cfg)[["item_id", "R"]]
    orig = pool[(pool["split"] == "pool") & ~pool["is_derived"]].merge(scores, on="item_id")
    rows = []
    for ds, g in orig.groupby("dataset", sort=True):
        g = g.copy()
        g["r_quintile"] = pd.qcut(g["R"].rank(method="first"), 5, labels=False).astype(int) + 1
        g["stratum"] = g["answer_type"] + "|q" + g["r_quintile"].astype(str)
        comp = (g["stratum"].value_counts(normalize=True)).to_dict()
        n = min(PER_DATASET, len(g))
        ids = set(sample_stratified(g.to_dict("records"), comp, n, "stratum", [cfg.global_seed, "sc_subsample", ds]))
        rows.append(g[g["item_id"].isin(ids)][["item_id", "dataset", "answer_type", "r_quintile"]])
    return pd.concat(rows).sort_values("item_id").reset_index(drop=True)


def run(cfg: DictConfig, llm: StrongClient | None = None) -> None:
    llm = llm or make_strong(cfg)
    sub = select_subsample(cfg)
    write_table("sc_subsample", sub, cfg)
    pool = read_table("pool", cfg).set_index("item_id")
    n_samples = int(cfg.model.strong.self_consistency.n_samples)
    sampling = sc_sampling(cfg)
    jobs = []
    for item_id in sub["item_id"]:
        item = pool.loc[item_id].to_dict() | {"item_id": item_id}
        prompt = build_messages(item, "reason")[-1]["content"]
        jobs += [(item_id, item, prompt, k) for k in range(n_samples)]

    def one(job):
        item_id, item, prompt, k = job
        seed = sc_seed(cfg, item_id, k)
        text = llm.generate_text(prompt, sampling, seed)
        answer = extract_answer(text, item["answer_type"])
        correct = answer is not None and answers_match(answer, gold_for_match(item), item["answer_type"])
        return {"item_id": item_id, "k": k, "seed": int(seed % 2**31),  # the seed the API receives
                "answer": answer, "correct": bool(correct),
                "text": text}

    with ThreadPoolExecutor(max_workers=int(cfg.model.strong.max_workers)) as ex:
        samples = pd.DataFrame(list(ex.map(one, jobs)))
    write_table("sc_samples", samples, cfg)

    rows = []
    for item_id, g in samples.groupby("item_id", sort=True):
        item = pool.loc[item_id]
        g = g.sort_values("k")
        modal, size = modal_group(list(g["answer"]), item["answer_type"])
        rows.append({"item_id": item_id, "n_samples": len(g), "n_extracted": int(g["answer"].notna().sum()),
                     "sc_acc": float(g["correct"].mean()), "sc_modal_share": size / len(g),
                     "modal_answer": modal,
                     "modal_correct": bool(modal is not None
                                           and answers_match(modal, gold_for_match(item.to_dict()), item["answer_type"])),
                     "generator_model": llm.model_version})
    write_table("self_consistency", pd.DataFrame(rows), cfg)
    out = pd.DataFrame(rows)
    print(f"self-consistency: {len(out)} items x {n_samples} samples; mean sc_acc {out['sc_acc'].mean():.3f}, "
          f"mean modal share {out['sc_modal_share'].mean():.3f}, unextracted samples {int((samples['answer'].isna()).sum())}")
