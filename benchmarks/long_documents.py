"""Long documents: is an attack found wherever it sits in long benign text, and what does
reading the whole document cost in false positives?

Usage:
    uv run --no-sync python benchmarks/long_documents.py [--docs 2000]

Builds documents from the test split only (no download): each document is one attack
prompt placed in benign padding, where the padding is benign test prompts drawn at random
and joined by blank lines until it reaches a target length. The attack goes at the start,
at the paragraph boundary nearest the middle, at a random interior paragraph boundary or
at the end; the same padding with no attack is the benign-only document. Attacks are test
attacks of at most 2,048 normalized characters (half a window), so each fits inside one
scoring window.

Every document is scored twice with the bundled model at its shipped threshold:

- ``head_tail``: ``max_windows=0``, the first and last 2,048 normalized characters only,
  which is how every text was scored before windowed scoring;
- ``windowed``: the default, the maximum over the head-and-tail view and overlapping
  4,096-character windows across the whole document.

Also measured:

- detection by attack length, because a short attack is diluted by the benign text that
  shares its window;
- an exploratory sweep over smaller windows (the same weights with a smaller
  ``max_chars``; not a shipped configuration), to show what a finer scan would catch and
  what it would cost in false positives;
- the same two scorings on the unmodified test split (the rows behind the README
  headline table), per source, and the threshold that would give 1% FPR on validation
  under windowed scoring;
- latency of both scorings on test prompts in their natural length mix, and the time to
  score one very long document.

Writes results/long_documents.json.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt
from common import PREPARED, environment, write_json

from prompt_injection_detector.data import Dataset
from prompt_injection_detector.features import MAX_WINDOWS
from prompt_injection_detector.metrics import (
    percentiles,
    rates_at,
    roc_auc,
    threshold_at_fpr,
    tpr_at_fpr,
)
from prompt_injection_detector.model import LinearDetector
from prompt_injection_detector.normalize import normalize

SEED = 0
PADDING_CHARS = (8_192, 16_384, 32_768, 65_536)
PRIMARY_PADDING = 16_384
MAX_PADDING_PROMPT_CHARS = 2_000
POSITIONS = ("start", "middle", "random", "end")
SEPARATOR = "\n\n"
# Attack length buckets in normalized characters: [low, high).
LENGTH_BUCKETS = ((1, 100), (100, 300), (300, 1000), (1000, 2049))
SWEEP_WINDOWS = (4096, 2048, 1024, 512, 256)
Scores = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Document:
    """One attack and the benign paragraphs around it."""

    attack: str
    paragraphs: tuple[str, ...]
    random_index: int

    def benign(self) -> str:
        return SEPARATOR.join(self.paragraphs)

    def middle_index(self) -> int:
        """The paragraph boundary closest to the middle of the padding, in characters."""
        total = sum(len(p) + len(SEPARATOR) for p in self.paragraphs)
        best, best_gap, offset = 1, float("inf"), 0
        for i, p in enumerate(self.paragraphs[:-1], start=1):
            offset += len(p) + len(SEPARATOR)
            if abs(offset - total / 2) < best_gap:
                best, best_gap = i, abs(offset - total / 2)
        return best

    def with_attack(self, position: str) -> str:
        at = {
            "start": 0,
            "middle": self.middle_index(),
            "random": self.random_index,
            "end": len(self.paragraphs),
        }[position]
        return SEPARATOR.join([*self.paragraphs[:at], self.attack, *self.paragraphs[at:]])


def outside_head_and_tail(text: str, attack: str, max_chars: int) -> bool:
    """Whether no character of ``attack`` is in the first or last ``max_chars // 2``
    normalized characters of ``text``, the part head-and-tail scoring reads."""
    s, a = normalize(text), normalize(attack)
    at = s.find(a)
    half = max_chars // 2
    return at >= half and at + len(a) <= len(s) - half


def sample_attacks(test: Dataset, n: int, max_chars: int, rng: random.Random) -> list[str]:
    """``n`` test attacks whose normalized form fits in half a window."""
    pool = [
        t
        for t, y in zip(test.text, test.label, strict=True)
        if y == 1 and 0 < len(normalize(t)) <= max_chars // 2
    ]
    return rng.sample(pool, n)


def build_documents(
    attacks: Sequence[str], benign_pool: Sequence[str], padding_chars: int, rng: random.Random
) -> list[Document]:
    """One document per attack, padded with distinct benign prompts to ``padding_chars``."""
    docs = []
    for attack in attacks:
        paragraphs: list[str] = []
        used: set[int] = set()
        total = 0
        while total < padding_chars:
            i = rng.randrange(len(benign_pool))
            if i in used:
                continue
            used.add(i)
            paragraphs.append(benign_pool[i])
            total += len(benign_pool[i]) + len(SEPARATOR)
        # An interior paragraph boundary: never the very start or the very end.
        docs.append(Document(attack, tuple(paragraphs), rng.randint(1, len(paragraphs) - 1)))
    return docs


def wilson(k: int, n: int) -> list[float]:
    """95% Wilson score interval for ``k`` successes in ``n`` trials."""
    if n == 0:
        return [0.0, 1.0]
    z = 1.959963984540054
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / (1 + z * z / n)
    return [max(0.0, centre - half), min(1.0, centre + half)]


def rate(scores: Scores, threshold: float) -> dict[str, object]:
    k, n = int((scores >= threshold).sum()), len(scores)
    return {"flagged": k, "n": n, "rate": k / n, "ci95": wilson(k, n)}


def both(
    head_tail: LinearDetector, windowed: LinearDetector, texts: list[str]
) -> dict[str, Scores]:
    return {"head_tail": head_tail.score_batch(texts), "windowed": windowed.score_batch(texts)}


def placement_results(
    head_tail: LinearDetector, windowed: LinearDetector, docs: list[Document], threshold: float
) -> dict[str, object]:
    """Detection by attack position and the benign-only false-positive rate."""
    out: dict[str, object] = {}
    lengths = [len(normalize(d.benign())) for d in docs]
    out["normalized_chars"] = {
        "median": float(np.median(lengths)),
        "min": int(min(lengths)),
        "max": int(max(lengths)),
    }
    out["paragraphs_median"] = float(np.median([len(d.paragraphs) for d in docs]))
    for position in POSITIONS:
        texts = [d.with_attack(position) for d in docs]
        s = both(head_tail, windowed, texts)
        row: dict[str, object] = {k: rate(v, threshold) for k, v in s.items()}
        row["attack_outside_head_and_tail"] = sum(
            outside_head_and_tail(t, d.attack, windowed.cfg.max_chars)
            for t, d in zip(texts, docs, strict=True)
        )
        out[f"attack_at_{position}"] = row
    s = both(head_tail, windowed, [d.benign() for d in docs])
    out["benign_only"] = {k: rate(v, threshold) for k, v in s.items()}
    return out


def by_attack_length(
    head_tail: LinearDetector, windowed: LinearDetector, docs: list[Document], threshold: float
) -> dict[str, object]:
    """Detection by attack position, split by the attack's normalized length."""
    n = np.asarray([len(normalize(d.attack)) for d in docs])
    out: dict[str, object] = {}
    scores = {
        position: both(head_tail, windowed, [d.with_attack(position) for d in docs])
        for position in POSITIONS
    }
    for lo, hi in LENGTH_BUCKETS:
        m = (n >= lo) & (n < hi)
        out[f"{lo}_to_{hi - 1}_chars"] = {
            "attacks": int(m.sum()),
            **{
                f"attack_at_{position}": {k: rate(v[m], threshold) for k, v in s.items()}
                for position, s in scores.items()
            },
        }
    return out


def window_sweep(
    windowed: LinearDetector, docs: list[Document], threshold: float
) -> dict[str, object]:
    """Exploratory: the same weights scanned with smaller windows. A text longer than the
    window is scored as the maximum over its head and tail (half a window each) and
    windows of that size at half-window stride."""
    middle = [d.with_attack("middle") for d in docs]
    benign = [d.benign() for d in docs]
    y = np.concatenate([np.ones(len(docs), dtype=np.int8), np.zeros(len(docs), dtype=np.int8)])
    out: dict[str, object] = {}
    for w in SWEEP_WINDOWS:
        det = LinearDetector(
            replace(windowed.cfg, max_chars=w),
            windowed.weights,
            windowed.idf,
            windowed.bias,
            windowed.threshold,
        )
        sm, sb = det.score_batch(middle), det.score_batch(benign)
        out[str(w)] = {
            "attack_at_middle": rate(sm, threshold),
            "benign_only": rate(sb, threshold),
            # Threshold free: detection when the threshold flags 1% of these benign documents.
            "attack_at_middle_tpr_at_1pct_document_fpr": tpr_at_fpr(
                y, np.concatenate([sm, sb]), 0.01
            ),
        }
        print("window", w, json.dumps(out[str(w)]), flush=True)
    return out


def split_metrics(y: npt.NDArray[np.int8], s: Scores, threshold: float) -> dict[str, float]:
    r = rates_at(y, s, threshold)
    return {
        "auroc": roc_auc(y, s),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, s, 0.01),
        "tpr_at_0.1pct_fpr": tpr_at_fpr(y, s, 0.001),
        "tpr_at_threshold": r.tpr,
        "fpr_at_threshold": r.fpr,
        "precision_at_threshold": r.precision,
    }


def test_split_effect(
    head_tail: LinearDetector, windowed: LinearDetector, test: Dataset, validation: Dataset
) -> dict[str, object]:
    """The two scorings on the unmodified test split, and a recalibrated threshold."""
    thr = windowed.threshold
    y = np.asarray(test.label, dtype=np.int8)
    long_row = np.asarray([len(normalize(t)) > windowed.cfg.max_chars for t in test.text])
    s = both(head_tail, windowed, test.text)
    out: dict[str, object] = {
        "rows": len(y),
        "attacks": int(y.sum()),
        "rows_over_the_limit": int(long_row.sum()),
        "attacks_over_the_limit": int((long_row & (y == 1)).sum()),
        "benign_over_the_limit": int((long_row & (y == 0)).sum()),
        "rows_with_a_changed_score": int((s["head_tail"] != s["windowed"]).sum()),
        "short_rows_with_a_changed_score": int((s["head_tail"] != s["windowed"])[~long_row].sum()),
    }
    for name, v in s.items():
        out[name] = split_metrics(y, v, thr)
        out[name]["benign_over_the_limit"] = rate(v[long_row & (y == 0)], thr)  # type: ignore[index]
        out[name]["attacks_over_the_limit"] = rate(v[long_row & (y == 1)], thr)  # type: ignore[index]
    # Sources whose flagged counts differ between the two scorings.
    src = np.asarray(test.source)
    changed: dict[str, object] = {}
    for name in sorted(set(src.tolist())):
        row = {}
        for label, key in ((1, "attacks_flagged"), (0, "benign_flagged")):
            m = (src == name) & (y == label)
            a, b = int((s["head_tail"][m] >= thr).sum()), int((s["windowed"][m] >= thr).sum())
            if a != b:
                row[key] = {"head_tail": a, "windowed": b, "n": int(m.sum())}
        if row:
            changed[name] = row
    out["sources_with_changed_counts"] = changed
    # What threshold would restore the 1% FPR rule on validation under windowed scoring?
    yv = np.asarray(validation.label, dtype=np.int8)
    sv = windowed.score_batch(validation.text)
    new_thr = threshold_at_fpr(sv[yv == 0], 0.01)
    r = rates_at(y, s["windowed"], new_thr)
    out["windowed_recalibrated"] = {
        "threshold": new_thr,
        "rule": "1% FPR on validation negatives, scored with windows",
        "validation_fpr_at_shipped_threshold": float((sv[yv == 0] >= thr).mean()),
        "tpr_at_threshold": r.tpr,
        "fpr_at_threshold": r.fpr,
    }
    return out


def natural_mix_latency(
    head_tail: LinearDetector, windowed: LinearDetector, test: Dataset
) -> dict[str, object]:
    """Both scorings on test prompts in their natural length mix: per call on the same
    10,000 prompts ``latency.py`` samples, and ``score_batch`` over the whole split."""
    sample = random.Random(0).sample(test.text, 10_000)
    out: dict[str, object] = {
        "unit": "microseconds per score() call; batch in texts per second, best of 3",
        "sample": len(sample),
        "sample_over_the_limit": sum(len(normalize(t)) > windowed.cfg.max_chars for t in sample),
    }
    for name, det in (("head_tail", head_tail), ("windowed", windowed)):
        for t in sample[:200]:
            det.score(t)
        times = []
        for t in sample:
            t0 = time.perf_counter_ns()
            det.score(t)
            times.append((time.perf_counter_ns() - t0) / 1e3)
        row = {k: round(v, 2) for k, v in percentiles(times).items()}
        row["mean"] = round(float(np.mean(times)), 2)
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            det.score_batch(test.text)
            best = min(best, time.perf_counter() - t0)
        row["batch_texts_per_second"] = round(len(test.text) / best)
        out[name] = row
    return out


def one_document_latency(
    head_tail: LinearDetector, windowed: LinearDetector, benign_pool: Sequence[str]
) -> dict[str, object]:
    """Milliseconds for one ``score`` call on one benign document of each size."""
    rng = random.Random(SEED)
    text = SEPARATOR.join(rng.sample(list(benign_pool), 4000))
    out: dict[str, object] = {"unit": "milliseconds per score() call, median of 7"}
    for chars in (16_384, 131_072, 1_048_576):
        doc = text[:chars]
        assert len(doc) == chars
        row: dict[str, float] = {"normalized_chars": len(normalize(doc))}
        for name, det in (("head_tail", head_tail), ("windowed", windowed)):
            det.score(doc)
            times = []
            for _ in range(7):
                t0 = time.perf_counter()
                det.score(doc)
                times.append((time.perf_counter() - t0) * 1e3)
            row[name] = round(statistics.median(times), 3)
        row["windows"] = len(windowed.view_margins(doc)) - 1
        out[f"{chars}_chars"] = row
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docs", type=int, default=2000, help="documents per padding length")
    args = ap.parse_args()

    windowed = LinearDetector.load_default()
    head_tail = windowed.with_max_windows(0)
    thr = windowed.threshold
    test = Dataset.load(PREPARED, split="test")
    validation = Dataset.load(PREPARED, split="validation")

    rng = random.Random(SEED)
    attacks = sample_attacks(test, args.docs, windowed.cfg.max_chars, rng)
    benign_pool = [
        t
        for t, y in zip(test.text, test.label, strict=True)
        if y == 0 and 20 <= len(t) <= MAX_PADDING_PROMPT_CHARS
    ]
    alone = windowed.score_batch(attacks)
    pool_scores = windowed.score_batch(benign_pool)
    res: dict[str, object] = {
        "model": "bundled linear-v1.npz",
        "threshold": thr,
        "max_chars": windowed.cfg.max_chars,
        "max_windows": {"head_tail": 0, "windowed": MAX_WINDOWS},
        "seed": SEED,
        "data": "test split of prepared.parquet; attacks of at most 2,048 normalized "
        "characters, padding from benign prompts of 20 to 2,000 characters",
        "documents_per_length": args.docs,
        "attack_pool": {
            "sampled": len(attacks),
            "normalized_chars_median": float(np.median([len(normalize(a)) for a in attacks])),
            "alone": rate(alone, thr),
        },
        "benign_pool": {"prompts": len(benign_pool), "alone": rate(pool_scores, thr)},
        "by_padding_chars": {},
    }
    for chars in PADDING_CHARS:
        docs = build_documents(attacks, benign_pool, chars, random.Random(SEED + chars))
        res["by_padding_chars"][str(chars)] = placement_results(head_tail, windowed, docs, thr)  # type: ignore[index]
        print(chars, json.dumps(res["by_padding_chars"][str(chars)]), flush=True)  # type: ignore[index]
    res["primary_padding_chars"] = PRIMARY_PADDING
    primary = build_documents(
        attacks, benign_pool, PRIMARY_PADDING, random.Random(SEED + PRIMARY_PADDING)
    )
    res["by_attack_length"] = by_attack_length(head_tail, windowed, primary, thr)
    res["window_size_sweep"] = {
        "note": "exploratory, not a shipped configuration: bundled weights with a smaller "
        "max_chars, at the shipped threshold, on the primary documents",
        "by_window_chars": window_sweep(windowed, primary, thr),
    }
    res["test_split"] = test_split_effect(head_tail, windowed, test, validation)
    res["natural_mix_latency"] = natural_mix_latency(head_tail, windowed, test)
    res["one_document_latency"] = one_document_latency(head_tail, windowed, benign_pool)
    res["environment"] = environment()
    write_json("long_documents.json", res)
    shown = ("by_attack_length", "test_split", "natural_mix_latency", "one_document_latency")
    print(json.dumps({k: res[k] for k in shown}, indent=2))


if __name__ == "__main__":
    main()
