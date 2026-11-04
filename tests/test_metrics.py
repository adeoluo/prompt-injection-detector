"""Metrics against brute-force definitions and, when installed, scikit-learn."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from prompt_injection_detector.metrics import (
    bootstrap_ci,
    percentiles,
    rates_at,
    roc_auc,
    roc_curve,
    threshold_at_fpr,
    tpr_at_fpr,
)

labelled = st.lists(
    st.tuples(st.integers(0, 1), st.sampled_from([0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])),
    min_size=2,
    max_size=60,
)


def brute_auc(y: list[int], s: list[float]) -> float:
    pos = [b for a, b in zip(y, s, strict=True) if a == 1]
    neg = [b for a, b in zip(y, s, strict=True) if a == 0]
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p, n in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


@settings(max_examples=200, deadline=None)
@given(labelled)
def test_auroc_matches_pairwise_definition(rows: list[tuple[int, float]]) -> None:
    y, s = zip(*rows, strict=True)
    assume(0 < sum(y) < len(y))
    assert roc_auc(y, s) == pytest.approx(brute_auc(list(y), list(s)))


@settings(max_examples=200, deadline=None)
@given(labelled, st.sampled_from([0.0, 0.01, 0.1, 0.3]))
def test_tpr_at_fpr_matches_threshold_scan(rows: list[tuple[int, float]], target: float) -> None:
    y, s = map(np.asarray, zip(*rows, strict=True))
    assume(0 < y.sum() < len(y))
    best = 0.0
    for t in [np.inf, *sorted(set(s.tolist()))]:
        r = rates_at(y, s, t)
        if r.fpr <= target + 1e-12:
            best = max(best, r.tpr)
    assert tpr_at_fpr(y, s, target) == pytest.approx(best)


@settings(max_examples=200, deadline=None)
@given(st.lists(st.floats(0, 1), min_size=1, max_size=200), st.floats(0, 0.5))
def test_threshold_at_fpr_respects_target(neg: list[float], target: float) -> None:
    t = threshold_at_fpr(neg, target)
    assert np.mean(np.asarray(neg) >= t) <= target + 1e-12


def test_roc_curve_endpoints() -> None:
    fpr, tpr, thr = roc_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    assert fpr[0] == 0 and tpr[0] == 0 and fpr[-1] == 1 and tpr[-1] == 1
    assert np.all(np.diff(thr[1:]) < 0)


def test_rates_and_edge_cases() -> None:
    r = rates_at([1, 1, 0, 0], [0.9, 0.2, 0.8, 0.1], 0.5)
    assert (r.tpr, r.fpr, r.precision) == (0.5, 0.5, 0.5)
    assert r.f1 == pytest.approx(0.5)
    assert rates_at([0, 0], [0.1, 0.2], 0.5).precision == 0.0
    with pytest.raises(ValueError):
        roc_auc([1, 1], [0.1, 0.2])
    with pytest.raises(ValueError):
        roc_auc([1, 2], [0.1, 0.2])
    with pytest.raises(ValueError):
        roc_auc([1, 0], [0.1])
    with pytest.raises(ValueError):
        threshold_at_fpr([], 0.1)
    with pytest.raises(ValueError):
        threshold_at_fpr([0.1], 1.0)


def test_bootstrap_ci_brackets_point() -> None:
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200), np.zeros(200)].astype(int)
    s = np.r_[rng.normal(1, 1, 200), rng.normal(0, 1, 200)]
    point, lo, hi = bootstrap_ci(roc_auc, y, s, n_boot=200)
    assert lo <= point <= hi
    assert 0.6 < point < 0.9


def test_percentiles() -> None:
    p = percentiles(range(1, 101))
    assert p["p50"] == pytest.approx(50.5)
    assert set(p) == {"p50", "p95", "p99"}


def test_matches_scikit_learn() -> None:
    sk = pytest.importorskip("sklearn.metrics")
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = np.round(rng.random(500), 2)
    assert roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s))
