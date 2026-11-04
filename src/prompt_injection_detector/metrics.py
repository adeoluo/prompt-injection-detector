"""Detection metrics: AUROC, ROC points, TPR at a fixed FPR, thresholds and bootstrap CIs.

All functions take label arrays (1 = injection) and scores where higher means more
likely an injection.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


def _arrays(y: Sequence[int] | npt.ArrayLike, s: Sequence[float] | npt.ArrayLike) -> tuple[
    npt.NDArray[np.int8], Array
]:  # fmt: skip
    ya = np.asarray(y, dtype=np.int8)
    sa = np.asarray(s, dtype=np.float64)
    if ya.shape != sa.shape or ya.ndim != 1:
        raise ValueError("labels and scores must be 1-D arrays of the same length")
    if not np.isin(ya, (0, 1)).all():
        raise ValueError("labels must be 0 or 1")
    return ya, sa


def roc_auc(y: npt.ArrayLike, s: npt.ArrayLike) -> float:
    """Area under the ROC curve (Mann-Whitney U with average ranks for ties)."""
    ya, sa = _arrays(y, s)
    n_pos = int(ya.sum())
    n_neg = len(ya) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUROC needs both classes")
    _, inv, counts = np.unique(sa, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    avg_rank = ends - (counts - 1) / 2.0
    ranks = avg_rank[inv]
    u = ranks[ya == 1].sum() - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def roc_curve(y: npt.ArrayLike, s: npt.ArrayLike) -> tuple[Array, Array, Array]:
    """``(fpr, tpr, thresholds)`` at every distinct score, highest threshold first."""
    ya, sa = _arrays(y, s)
    order = np.argsort(-sa, kind="mergesort")
    ss, yy = sa[order], ya[order]
    last = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tps = np.cumsum(yy)[last]
    fps = (last + 1) - tps
    n_pos, n_neg = max(int(ya.sum()), 1), max(len(ya) - int(ya.sum()), 1)
    return (
        np.r_[0.0, fps / n_neg],
        np.r_[0.0, tps / n_pos],
        np.r_[np.inf, ss[last]],
    )


def tpr_at_fpr(y: npt.ArrayLike, s: npt.ArrayLike, target: float) -> float:
    """Highest TPR among operating points whose FPR is at most ``target``."""
    fpr, tpr, _ = roc_curve(y, s)
    return float(np.max(tpr[fpr <= target + 1e-12]))


def threshold_at_fpr(negative_scores: npt.ArrayLike, target: float) -> float:
    """Smallest threshold ``t`` with ``mean(neg >= t) <= target``."""
    neg = np.sort(np.asarray(negative_scores, dtype=np.float64))[::-1]
    if neg.size == 0:
        raise ValueError("need at least one negative score")
    if not 0.0 <= target < 1.0:
        raise ValueError("target must be in [0, 1)")
    k = int(np.floor(target * neg.size + 1e-9))
    return float(np.nextafter(neg[k], np.inf))


@dataclass(frozen=True)
class Rates:
    """Confusion-matrix rates at one threshold."""

    threshold: float
    tpr: float
    fpr: float
    precision: float
    f1: float
    n_pos: int
    n_neg: int


def rates_at(y: npt.ArrayLike, s: npt.ArrayLike, threshold: float) -> Rates:
    ya, sa = _arrays(y, s)
    pred = sa >= threshold
    tp = int((pred & (ya == 1)).sum())
    fp = int((pred & (ya == 0)).sum())
    n_pos = int(ya.sum())
    n_neg = len(ya) - n_pos
    tpr = tp / n_pos if n_pos else float("nan")
    fpr = fp / n_neg if n_neg else float("nan")
    precision = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * precision * tpr / (precision + tpr) if n_pos and precision + tpr else 0.0
    return Rates(threshold, tpr, fpr, precision, f1, n_pos, n_neg)


def bootstrap_ci(
    metric: Callable[[npt.NDArray[np.int8], Array], float],
    y: npt.ArrayLike,
    s: npt.ArrayLike,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """``(point, low, high)``: percentile bootstrap CI, resampling each class separately."""
    ya, sa = _arrays(y, s)
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(ya == 1), np.flatnonzero(ya == 0)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.r_[rng.choice(pos, pos.size), rng.choice(neg, neg.size)]
        vals[b] = metric(ya[idx], sa[idx])
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return metric(ya, sa), float(lo), float(hi)


def percentiles(samples: npt.ArrayLike, qs: Sequence[float] = (50, 95, 99)) -> dict[str, float]:
    """Named percentiles (``p50`` and so on) of latency samples."""
    a = np.asarray(samples, dtype=np.float64)
    return {f"p{int(q)}": float(np.percentile(a, q)) for q in qs}
