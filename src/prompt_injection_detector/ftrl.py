"""FTRL-Proximal logistic regression (McMahan et al., 2013), reference implementation.

The Rust core runs the same algorithm, in the same seeded example order, with
featurization parallelized across threads; the parity tests check that both produce
the same weights. Coordinate ``dim`` is the bias.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from prompt_injection_detector.features import (
    FeatureConfig,
    bucket_counts_reference,
    idf_from_df,
    tfidf_reference,
)
from prompt_injection_detector.rng import permutation_reference


@dataclass(frozen=True)
class FtrlConfig:
    """FTRL-Proximal hyperparameters."""

    alpha: float = 0.5
    beta: float = 1.0
    l1: float = 0.0
    l2: float = 1e-4
    epochs: int = 3
    seed: int = 13
    pos_weight: float = 1.0

    def __post_init__(self) -> None:
        if self.alpha <= 0 or self.beta < 0 or self.l1 < 0 or self.l2 < 0:
            raise ValueError("alpha must be positive and beta, l1, l2 non-negative")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1")
        if self.pos_weight <= 0:
            raise ValueError("pos_weight must be positive")

    def as_dict(self) -> dict[str, float | int]:
        return asdict(self)


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))


class Ftrl:
    """Per-coordinate FTRL-Proximal state."""

    def __init__(self, dim: int, cfg: FtrlConfig) -> None:
        self.cfg = cfg
        self.z = [0.0] * (dim + 1)
        self.n = [0.0] * (dim + 1)

    def weight(self, i: int) -> float:
        z = self.z[i]
        c = self.cfg
        if abs(z) <= c.l1:
            return 0.0
        sign = -1.0 if z < 0.0 else 1.0
        return -(z - sign * c.l1) / ((c.beta + math.sqrt(self.n[i])) / c.alpha + c.l2)

    def update(self, idx: Sequence[int], val: Sequence[float], label: float) -> float:
        bias = len(self.z) - 1
        w = [self.weight(i) for i in idx]
        z = 0.0
        for wi, v in zip(w, val, strict=True):
            z += wi * v
        wb = self.weight(bias)
        z += wb
        p = sigmoid(z)
        g = (p - label) * (self.cfg.pos_weight if label > 0.5 else 1.0)
        a = self.cfg.alpha
        for i, v, wi in zip(idx, val, w, strict=True):
            gi = g * v
            ni = self.n[i]
            sigma = (math.sqrt(ni + gi * gi) - math.sqrt(ni)) / a
            self.z[i] += gi - sigma * wi
            self.n[i] = ni + gi * gi
        nb = self.n[bias]
        sigma = (math.sqrt(nb + g * g) - math.sqrt(nb)) / a
        self.z[bias] += g - sigma * wb
        self.n[bias] = nb + g * g
        return p

    def finish(self) -> tuple[list[float], float]:
        dim = len(self.z) - 1
        return [self.weight(i) for i in range(dim)], self.weight(dim)


def train_reference(
    texts: Sequence[str],
    labels: Sequence[float],
    fcfg: FeatureConfig,
    tcfg: FtrlConfig,
) -> tuple[list[float], list[float], float, list[float]]:
    """Train on ``texts``; returns ``(weights, idf, bias, epoch_losses)``."""
    if len(texts) != len(labels):
        raise ValueError("texts and labels differ in length")
    if not texts:
        raise ValueError("no training examples")
    feats = [bucket_counts_reference(t, fcfg) for t in texts]
    df = [0] * fcfg.dim
    for idx, _ in feats:
        for i in idx:
            df[i] += 1
    idf = idf_from_df(df, len(texts))
    vecs = [tfidf_reference(idx, cnt, idf) for idx, cnt in feats]
    ftrl = Ftrl(fcfg.dim, tcfg)
    losses = []
    for epoch in range(tcfg.epochs):
        loss = 0.0
        for k in permutation_reference(len(texts), (tcfg.seed + epoch) & ((1 << 64) - 1)):
            y = float(labels[k])
            p = ftrl.update(vecs[k][0], vecs[k][1], y)
            loss -= y * math.log(max(p, 1e-15)) + (1.0 - y) * math.log(max(1.0 - p, 1e-15))
        losses.append(loss / len(texts))
    weights, bias = ftrl.finish()
    return weights, idf, bias, losses
