"""SplitMix64, the seeded generator shared with the Rust core."""

from __future__ import annotations

MASK64 = (1 << 64) - 1


def mix64(z: int) -> int:
    """The SplitMix64 output finalizer."""
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)


class SplitMix64:
    """SplitMix64 (Steele, Lea and Flood, 2014)."""

    def __init__(self, seed: int) -> None:
        self.state = seed & MASK64

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & MASK64
        return mix64(self.state)


def permutation_reference(n: int, seed: int) -> list[int]:
    """Fisher-Yates permutation of ``range(n)`` driven by SplitMix64."""
    p = list(range(n))
    rng = SplitMix64(seed)
    for i in range(n - 1, 0, -1):
        j = rng.next_u64() % (i + 1)
        p[i], p[j] = p[j], p[i]
    return p
