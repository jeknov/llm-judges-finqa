"""Deduplication and leakage removal (Spec 1, Deduplication).

1. Exact: drop items whose normalised question + context hash matches an earlier item (first by item_id).
2. Leakage: drop pool items with Jaccard >= threshold on question + context against any probe or test item.
   Shingles are word n-grams of lowercased whitespace tokens (P6: 5-grams, 128 permutations). MinHash LSH
   proposes candidate pairs and every candidate is confirmed with the exact Jaccard of the shingle sets,
   so the decision never rests on a MinHash estimate. datasketch's default MinHash seed is used.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from datasketch import MinHash, MinHashLSH


def item_text(item: Mapping) -> str:
    cells = " ".join(c for row in item["table"] for c in row)
    return " ".join([item["question"], *item["context_text"], cells])


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def shingles(text: str, n: int) -> set[str]:
    words = normalize(text).split()
    if len(words) <= n:
        return {" ".join(words)}
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 1.0


def exact_duplicates(items: Sequence[Mapping]) -> list[tuple[str, str]]:
    """(dropped_id, kept_id) pairs; items are scanned in item_id order and the first is kept."""
    seen: dict[str, str] = {}
    dropped = []
    for item in sorted(items, key=lambda it: it["item_id"]):
        key = hashlib.sha256(normalize(item_text(item)).encode()).hexdigest()
        if key in seen:
            dropped.append((item["item_id"], seen[key]))
        else:
            seen[key] = item["item_id"]
    return dropped


def _minhash(sh: set[str], num_perm: int) -> MinHash:
    m = MinHash(num_perm=num_perm)
    m.update_batch([s.encode() for s in sorted(sh)])
    return m


def leakage(
    pool: Sequence[Mapping], others: Sequence[Mapping], threshold: float, num_perm: int, n: int
) -> list[tuple[str, str, float]]:
    """(pool_id, other_id, jaccard) for every pool item at or above threshold with some other item."""
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    other_sh = {}
    for item in others:
        sh = shingles(item_text(item), n)
        other_sh[item["item_id"]] = sh
        lsh.insert(item["item_id"], _minhash(sh, num_perm))
    pairs = []
    for item in sorted(pool, key=lambda it: it["item_id"]):
        sh = shingles(item_text(item), n)
        best = None
        for other_id in sorted(lsh.query(_minhash(sh, num_perm))):
            j = jaccard(sh, other_sh[other_id])
            if j >= threshold and (best is None or j > best[1]):
                best = (other_id, j)
        if best is not None:
            pairs.append((item["item_id"], best[0], best[1]))
    return pairs
