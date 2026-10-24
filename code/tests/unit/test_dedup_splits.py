"""Deduplication, leakage removal and stratified sampling (Spec 1, Deduplication; Probe and test sets)."""

from __future__ import annotations

import random

import pytest

from fqsel.data.dedup import exact_duplicates, jaccard, leakage, shingles
from fqsel.data.splits import SamplingError, allocate, sample_stratified


def _item(item_id, question, context, table=()):
    return {"item_id": item_id, "question": question, "context_text": context, "table": [list(r) for r in table]}


def _words(seed, n=80):
    r = random.Random(seed)
    return " ".join(r.choice("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu".split())
                    + str(r.randint(0, 999)) for _ in range(n))


def test_exact_duplicates_keep_first_by_id():
    a = _item("finqa:b", "What is X?", ["Same  context ."])
    b = _item("finqa:a", "what is x?", ["same context ."])  # identical after normalisation
    c = _item("finqa:c", "What is Y?", ["Same context ."])
    assert exact_duplicates([a, b, c]) == [("finqa:b", "finqa:a")]


def test_leakage_planted_near_duplicate():
    base = _words(1, 200)
    near = base + " extra"                     # Jaccard just below 1
    far = _words(2, 200)
    pool = [_item("p:near", "q", [near]), _item("p:far", "q", [far])]
    others = [_item("t:1", "q", [base])]
    pairs = leakage(pool, others, threshold=0.9, num_perm=128, n=5)
    assert [(p, o) for p, o, _ in pairs] == [("p:near", "t:1")]
    assert pairs[0][2] >= 0.9


def test_leakage_threshold_uses_exact_jaccard():
    base = _words(3, 100).split()
    edited = list(base)
    for i in range(0, 100, 12):                # change ~9 words: exact Jaccard falls well below 0.9
        edited[i] = "changed"
    a, b = " ".join(base), " ".join(edited)
    j = jaccard(shingles("q " + a, 5), shingles("q " + b, 5))
    assert j < 0.9
    assert leakage([_item("p", "q", [b])], [_item("t", "q", [a])], 0.9, 128, 5) == []


def test_shingles_short_text():
    assert shingles("Only three words", 5) == {"only three words"}


def test_allocate_largest_remainder():
    assert allocate({"arith": 70, "lookup": 20}, 20) == {"arith": 16, "lookup": 4}
    assert allocate({"a": 1, "b": 1, "c": 1}, 4) == {"a": 2, "b": 1, "c": 1}  # ties by name
    assert sum(allocate({"x": 3, "y": 7, "z": 11}, 13).values()) == 13


def test_sample_stratified_deterministic_and_proportional():
    cands = [{"item_id": f"i{k:03d}", "answer_type": "arith" if k < 80 else "lookup"} for k in range(100)]
    a = sample_stratified(cands, {"arith": 4, "lookup": 1}, 10, "answer_type", [1, "probe", "finqa"])
    b = sample_stratified(list(reversed(cands)), {"arith": 4, "lookup": 1}, 10, "answer_type", [1, "probe", "finqa"])
    assert a == b and len(a) == 10
    assert sum(1 for i in a if int(i[1:]) >= 80) == 2
    c = sample_stratified(cands, {"arith": 4, "lookup": 1}, 10, "answer_type", [1, "test", "finqa"])
    assert a != c


def test_sample_stratified_raises_when_short():
    cands = [{"item_id": "i1", "answer_type": "lookup"}]
    with pytest.raises(SamplingError):
        sample_stratified(cands, {"lookup": 1}, 2, "answer_type", [1])
