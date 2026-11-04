"""Near-duplicate grouping with MinHash and banded LSH for leakage-safe splits.

Texts in one group always land in the same split, so a template and its light
rewrites cannot appear in both training and test data. Reference implementation of
the Rust ``near_duplicate_groups``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from prompt_injection_detector._backend import Backend, resolve, rust
from prompt_injection_detector.features import fnv1a, words
from prompt_injection_detector.normalize import normalize
from prompt_injection_detector.rng import SplitMix64, mix64


@dataclass(frozen=True)
class DedupConfig:
    """MinHash and LSH settings."""

    num_perm: int = 64
    bands: int = 16
    shingle: int = 3
    threshold: float = 0.7
    seed: int = 1

    def __post_init__(self) -> None:
        if self.bands < 1 or self.num_perm < 1 or self.num_perm % self.bands:
            raise ValueError("num_perm must be a positive multiple of bands")
        if self.shingle < 1:
            raise ValueError("shingle must be positive")


def _shingle_hashes(text: str, k: int) -> list[int]:
    ws = words(normalize(text, backend="python"))

    def join(part: Sequence[str]) -> int:
        return fnv1a(ord("s"), b"\x00".join(w.encode() for w in part))

    if len(ws) < k:
        return [join(ws)]
    return [join(ws[i : i + k]) for i in range(len(ws) - k + 1)]


def perm_seeds(num_perm: int, seed: int) -> list[int]:
    rng = SplitMix64(seed)
    return [rng.next_u64() for _ in range(num_perm)]


def signature_reference(text: str, num_perm: int, shingle: int, seed: int) -> list[int]:
    """32-bit MinHash signature (reference)."""
    hs = _shingle_hashes(text, shingle)
    return [min(mix64(h ^ s) for h in hs) >> 32 for s in perm_seeds(num_perm, seed)]


def _find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def near_duplicate_groups_reference(texts: Sequence[str], cfg: DedupConfig) -> list[int]:
    """Group id (smallest member index) of every text (reference)."""
    sigs = [signature_reference(t, cfg.num_perm, cfg.shingle, cfg.seed) for t in texts]
    rows = cfg.num_perm // cfg.bands
    need = math.ceil(cfg.threshold * cfg.num_perm)
    parent = list(range(len(texts)))
    for b in range(cfg.bands):
        first: dict[tuple[int, ...], int] = {}
        for d, sig in enumerate(sigs):
            key = tuple(sig[b * rows : (b + 1) * rows])
            f = first.get(key)
            if f is None:
                first[key] = d
            elif sum(x == y for x, y in zip(sig, sigs[f], strict=True)) >= need:
                ra, rb = _find(parent, d), _find(parent, f)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    return [_find(parent, i) for i in range(len(texts))]


def near_duplicate_groups(
    texts: Sequence[str], cfg: DedupConfig | None = None, backend: Backend = "auto"
) -> list[int]:
    """Group id of every text; near-duplicates share an id."""
    cfg = cfg or DedupConfig()
    if resolve(backend) == "rust":
        return list(
            rust().near_duplicate_groups(
                list(texts), cfg.num_perm, cfg.bands, cfg.shingle, cfg.threshold, cfg.seed
            )
        )
    return near_duplicate_groups_reference(texts, cfg)
