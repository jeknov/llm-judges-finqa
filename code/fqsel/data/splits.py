"""Stratified sampling of probe, fixed-demo, test and (tiny) pool sets (Spec 1, Probe and test sets).

Targets are allocated across strata in proportion to the pool's composition by the largest-remainder
method (ties broken by stratum name). Within a stratum, candidate IDs are sorted and permuted with a
stable_seed derived from the global seed, the set's name, the dataset and the stratum.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence

from fqsel.ids import rng


class SamplingError(RuntimeError):
    pass


def allocate(proportions: Mapping[str, float], total: int) -> dict[str, int]:
    """Largest-remainder allocation of `total` in proportion to `proportions` (any non-negative weights)."""
    weight = sum(proportions.values())
    if weight <= 0:
        raise SamplingError("cannot allocate with zero total weight")
    exact = {k: total * v / weight for k, v in proportions.items()}
    alloc = {k: math.floor(x) for k, x in exact.items()}
    remainder = total - sum(alloc.values())
    order = sorted(exact, key=lambda k: (-(exact[k] - alloc[k]), k))
    for k in order[:remainder]:
        alloc[k] += 1
    return alloc


def sample_stratified(
    candidates: Sequence[Mapping],
    proportions: Mapping[str, float],
    total: int,
    key: str,
    seed_parts: Sequence,
) -> list[str]:
    """IDs of `total` candidates, stratified on candidates[key] in proportion to `proportions`."""
    alloc = allocate(proportions, total)
    by_stratum: dict[str, list[str]] = {}
    for c in candidates:
        by_stratum.setdefault(c[key], []).append(c["item_id"])
    chosen = []
    for stratum in sorted(alloc):
        k = alloc[stratum]
        ids = sorted(by_stratum.get(stratum, []))
        if len(ids) < k:
            raise SamplingError(
                f"{seed_parts}: stratum {stratum!r} needs {k} items but only {len(ids)} survive exclusions"
            )
        order = rng(*seed_parts, stratum).permutation(len(ids))
        chosen.extend(ids[i] for i in order[:k])
    return sorted(chosen)


def composition(items: Sequence[Mapping], key: str) -> dict[str, float]:
    return dict(Counter(it[key] for it in items))
