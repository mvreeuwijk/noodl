"""Read tracking for `Model.check(..., probe=True)` (see `noodl.validation`).

`TrackedDict` records which keys are READ. The two helpers let a site that handles every
driver in bulk -- the differentiable potential solve threads all of them through autograd
-- do so without marking them all as read, and keep tracking the dictionary it rebuilds,
so that the reads its elements and drives make are still recorded. Outside a probe both
helpers are exactly `[m[k] for k in keys]` and the identity.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class TrackedDict(dict):
    """A dict that records which keys are read (`[]`, `get`, `in`)."""

    def __init__(self, data, reads: set) -> None:
        super().__init__(data)
        self._reads = reads

    def __getitem__(self, key):
        self._reads.add(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        self._reads.add(key)
        return super().get(key, default)

    def __contains__(self, key) -> bool:
        self._reads.add(key)
        return super().__contains__(key)


def bulk_values(mapping: Mapping, keys: Sequence) -> list:
    """`[mapping[k] for k in keys]`, not recorded as reads."""
    if isinstance(mapping, TrackedDict):
        return [dict.__getitem__(mapping, k) for k in keys]
    return [mapping[k] for k in keys]


def same_tracking(original: Mapping, rebuilt: dict) -> dict:
    """`rebuilt`, recording reads into `original`'s record when `original` is tracked."""
    if isinstance(original, TrackedDict):
        return TrackedDict(rebuilt, original._reads)
    return rebuilt
