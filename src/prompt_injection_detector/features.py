"""Hashed character and word n-gram features with sublinear TF-IDF weighting.

Reference implementation of the Rust featurizer. For a normalized text ``s``:

- character n-grams for n in ``char_min..char_max`` over ``" " + s + " "``;
- word unigrams and bigrams, where a word is a maximal run of characters that are
  neither a space nor ASCII punctuation.

Each n-gram is hashed with 64-bit FNV-1a (with a one-byte namespace prefix) into
``dim`` buckets. A bucket's value is ``(1 + ln(count)) * idf[bucket]`` and the vector
is L2-normalized.

A normalized text longer than ``max_chars`` has two readings. Training featurizes its
head and tail only (``truncate``). Scoring featurizes several views (``views``): the same
head-and-tail text, plus overlapping windows of ``max_chars`` characters that cover the
text, so an instruction placed in the middle of a long document is still read.
"""

from __future__ import annotations

import itertools
import math
import string
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass

from prompt_injection_detector._backend import Backend, resolve, rust
from prompt_injection_detector.normalize import normalize

FNV_OFFSET = 0xCBF29CE484222325
FNV_PRIME = 0x100000001B3
_MASK64 = (1 << 64) - 1
_PUNCT = frozenset(string.punctuation)

#: Default cap on the number of windows scored for one long text. With the default
#: ``max_chars`` of 4,096 (stride 2,048) the windows cover the first 1,050,624 normalized
#: characters; see ``window_starts`` for what happens beyond the cap.
MAX_WINDOWS = 512


@dataclass(frozen=True)
class FeatureConfig:
    """Feature extraction settings shared by training and scoring."""

    dim: int = 1 << 20
    char_min: int = 3
    char_max: int = 5
    word_bigrams: bool = True
    max_chars: int = 4096

    def __post_init__(self) -> None:
        if self.dim < 2 or self.dim & (self.dim - 1) or self.dim > 1 << 30:
            raise ValueError(f"dim must be a power of two in [2, 2^30], got {self.dim}")
        if not 1 <= self.char_min <= self.char_max:
            raise ValueError("need 1 <= char_min <= char_max")
        if self.max_chars < 2:
            raise ValueError("max_chars must be at least 2")

    def as_dict(self) -> dict[str, int | bool]:
        return asdict(self)

    def args(self) -> tuple[int, int, int, bool, int]:
        """Positional arguments for the Rust functions."""
        return (self.dim, self.char_min, self.char_max, self.word_bigrams, self.max_chars)


def fnv1a(prefix: int, data: bytes) -> int:
    """64-bit FNV-1a over ``prefix`` followed by ``data``."""
    h = ((FNV_OFFSET ^ prefix) * FNV_PRIME) & _MASK64
    for b in data:
        h = ((h ^ b) * FNV_PRIME) & _MASK64
    return h


def bucket(h: int, dim: int) -> int:
    return (h ^ (h >> 32)) & (dim - 1)


def truncate(s: str, max_chars: int) -> str:
    """Keep the first and last ``max_chars // 2`` characters of a long text."""
    if len(s) <= max_chars:
        return s
    half = max_chars // 2
    return f"{s[:half]} {s[len(s) - half :]}"


def window_starts(n_chars: int, max_chars: int, max_windows: int = MAX_WINDOWS) -> list[int]:
    """Start offsets (in characters) of the scoring windows of a normalized text.

    Empty when the text fits in ``max_chars``. Windows are ``max_chars`` long and start
    every ``max_chars // 2`` characters; the last one is aligned to the end of the text,
    so every window is full and any span of up to ``max_chars - max_chars // 2``
    characters lies inside at least one window. At most ``max_windows`` windows are
    returned: a text that needs more keeps the first ``max_windows`` grid positions, and
    the characters after the last of them are not covered by any window.
    """
    if max_chars < 2:
        raise ValueError("max_chars must be at least 2")
    if max_windows < 0:
        raise ValueError("max_windows must not be negative")
    out: list[int] = []
    if n_chars <= max_chars:
        return out
    stride = max(max_chars // 2, 1)
    start = 0
    while len(out) < max_windows:
        if start + max_chars >= n_chars:
            out.append(n_chars - max_chars)
            break
        out.append(start)
        start += stride
    return out


def views(s: str, max_chars: int, max_windows: int = MAX_WINDOWS) -> list[str]:
    """The texts a scorer featurizes for one normalized text ``s``.

    A text of at most ``max_chars`` characters is its own single view. A longer text
    gives the head-and-tail text of ``truncate`` first, then one view per window of
    ``window_starts``. The score of the text is the highest score of any view.
    """
    if len(s) <= max_chars:
        return [s]
    starts = window_starts(len(s), max_chars, max_windows)
    return [truncate(s, max_chars), *(s[a : a + max_chars] for a in starts)]


def words(s: str) -> list[str]:
    """Maximal runs of characters that are neither a space nor ASCII punctuation."""
    out: list[str] = []
    cur: list[str] = []
    for c in s:
        if c == " " or c in _PUNCT:
            if cur:
                out.append("".join(cur))
                cur = []
        else:
            cur.append(c)
    if cur:
        out.append("".join(cur))
    return out


def ngrams(s: str, cfg: FeatureConfig) -> Iterator[tuple[str, str]]:
    """``(kind, ngram)`` pairs of a normalized text: kind is char, word or bigram.

    A text longer than ``cfg.max_chars`` is reduced to its head and tail first; this is
    the featurization used in training.
    """
    return view_ngrams(truncate(s, cfg.max_chars), cfg)


def view_ngrams(s: str, cfg: FeatureConfig) -> Iterator[tuple[str, str]]:
    """``(kind, ngram)`` pairs of one view (see ``views``), with no truncation."""
    padded = f" {s} "
    for n in range(cfg.char_min, cfg.char_max + 1):
        for i in range(len(padded) - n + 1):
            yield "char", padded[i : i + n]
    ws = words(s)
    for w in ws:
        yield "word", w
    if cfg.word_bigrams:
        for a, b in itertools.pairwise(ws):
            yield "bigram", f"{a} {b}"


def ngram_bucket(kind: str, gram: str, dim: int) -> int:
    if kind == "char":
        return bucket(fnv1a(ord("c"), gram.encode()), dim)
    if kind == "word":
        return bucket(fnv1a(ord("w"), gram.encode()), dim)
    a, b = gram.split(" ", 1)
    return bucket(fnv1a(ord("b"), a.encode() + b"\x00" + b.encode()), dim)


def view_bucket_counts(view: str, cfg: FeatureConfig) -> tuple[list[int], list[int]]:
    """Sorted ``(buckets, counts)`` of one view of a normalized text (reference)."""
    tally: dict[int, int] = {}
    for kind, gram in view_ngrams(view, cfg):
        b = ngram_bucket(kind, gram, cfg.dim)
        tally[b] = tally.get(b, 0) + 1
    keys = sorted(tally)
    return keys, [tally[k] for k in keys]


def bucket_counts_reference(text: str, cfg: FeatureConfig) -> tuple[list[int], list[int]]:
    """Sorted ``(buckets, counts)`` of a raw text (reference).

    A long text is counted on its head and tail only, as in training.
    """
    return view_bucket_counts(truncate(normalize(text, backend="python"), cfg.max_chars), cfg)


def bucket_counts(
    text: str, cfg: FeatureConfig, backend: Backend = "auto"
) -> tuple[list[int], list[int]]:
    """Sorted ``(buckets, counts)`` of the hashed n-grams of ``text``.

    A long text is counted on its head and tail only, as in training.
    """
    if resolve(backend) == "rust":
        idx, cnt = rust().bucket_counts(text, *cfg.args())
        return list(idx), list(cnt)
    return bucket_counts_reference(text, cfg)


def tfidf_reference(
    idx: Sequence[int], cnt: Sequence[int], idf: Sequence[float]
) -> tuple[list[int], list[float]]:
    """L2-normalized sublinear TF-IDF values for sorted bucket counts."""
    vals = [(1.0 + math.log(c)) * idf[i] for i, c in zip(idx, cnt, strict=True)]
    sq = 0.0
    for v in vals:
        sq += v * v
    if sq > 0.0:
        norm = math.sqrt(sq)
        vals = [v / norm for v in vals]
    return list(idx), vals


def idf_from_df(df: Sequence[int], n_docs: int) -> list[float]:
    """Smoothed inverse document frequency ``ln((1 + n) / (1 + df)) + 1``."""
    return [math.log((1.0 + n_docs) / (1.0 + d)) + 1.0 for d in df]
