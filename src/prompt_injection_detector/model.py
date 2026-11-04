"""The linear detector: hashed n-gram TF-IDF features scored by a logistic model.

Training runs FTRL-Proximal in the Rust core (or the Python reference). A trained
model is a small ``.npz`` file holding float32 weights and IDF values, the bias, the
feature settings and the decision threshold.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from prompt_injection_detector._backend import Backend, resolve, rust
from prompt_injection_detector.features import (
    FeatureConfig,
    bucket_counts,
    ngram_bucket,
    ngrams,
    tfidf_reference,
)
from prompt_injection_detector.ftrl import FtrlConfig, train_reference
from prompt_injection_detector.normalize import normalize

DEFAULT_MODEL = "linear-v1.npz"


@dataclass(frozen=True)
class Contribution:
    """One n-gram's share of a score (positive pushes towards injection)."""

    kind: str
    ngram: str
    weight: float


@dataclass
class LinearDetector:
    """A trained hashed n-gram logistic model."""

    cfg: FeatureConfig
    weights: npt.NDArray[np.float32]
    idf: npt.NDArray[np.float32]
    bias: float
    threshold: float = 0.5
    meta: dict[str, Any] = field(default_factory=dict)
    backend: Backend = "auto"

    def __post_init__(self) -> None:
        self.weights = np.ascontiguousarray(self.weights, dtype="<f4")
        self.idf = np.ascontiguousarray(self.idf, dtype="<f4")
        if self.weights.shape != (self.cfg.dim,) or self.idf.shape != (self.cfg.dim,):
            raise ValueError("weights and idf must have cfg.dim entries")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        self._impl = resolve(self.backend)
        self._scorer: Any = None
        if self._impl == "rust":
            self._scorer = rust().LinearScorer(
                self.weights.tobytes(), self.idf.tobytes(), float(self.bias), *self.cfg.args()
            )
        self._w64 = self.weights.astype(np.float64)
        self._idf64 = self.idf.astype(np.float64)

    # ------------------------------------------------------------------ training
    @classmethod
    def train(
        cls,
        texts: Sequence[str],
        labels: Sequence[int],
        cfg: FeatureConfig | None = None,
        ftrl: FtrlConfig | None = None,
        backend: Backend = "auto",
    ) -> LinearDetector:
        """Fit on ``texts`` with binary ``labels`` (1 = injection or jailbreak)."""
        cfg = cfg or FeatureConfig()
        ftrl = ftrl or FtrlConfig()
        y = [float(v) for v in labels]
        if any(v not in (0.0, 1.0) for v in y):
            raise ValueError("labels must be 0 or 1")
        if resolve(backend) == "rust":
            w_b, idf_b, bias, losses = rust().train_ftrl(
                list(texts),
                y,
                *cfg.args(),
                ftrl.alpha,
                ftrl.beta,
                ftrl.l1,
                ftrl.l2,
                ftrl.epochs,
                ftrl.seed,
                ftrl.pos_weight,
            )
            weights = np.frombuffer(w_b, dtype="<f8")
            idf = np.frombuffer(idf_b, dtype="<f8")
        else:
            w_l, idf_l, bias, losses = train_reference(texts, y, cfg, ftrl)
            weights, idf = np.asarray(w_l), np.asarray(idf_l)
        meta = {
            "ftrl": ftrl.as_dict(),
            "epoch_losses": [float(v) for v in losses],
            "n_train": len(y),
            "n_positive": int(sum(y)),
        }
        return cls(cfg, weights.astype("<f4"), idf.astype("<f4"), float(bias), meta=meta,
                   backend=backend)  # fmt: skip

    # ------------------------------------------------------------------- scoring
    def _features(self, text: str) -> tuple[list[int], list[float]]:
        idx, cnt = bucket_counts(text, self.cfg, backend=self._impl)
        return tfidf_reference(idx, cnt, self._idf64)  # type: ignore[arg-type]

    def margin(self, text: str) -> float:
        """Raw logit ``bias + w . x``."""
        if self._scorer is not None:
            return float(self._scorer.margin(text))
        idx, val = self._features(text)
        z = float(self.bias)
        for i, v in zip(idx, val, strict=True):
            z += float(self._w64[i]) * v
        return z

    def score(self, text: str) -> float:
        """Probability that ``text`` is a prompt injection or jailbreak."""
        if self._scorer is not None:
            return float(self._scorer.score(text))
        return 1.0 / (1.0 + math.exp(-self.margin(text)))

    def score_batch(self, texts: Sequence[str]) -> npt.NDArray[np.float64]:
        """Probabilities for many texts (parallel in the Rust core)."""
        if self._scorer is not None:
            return np.asarray(self._scorer.score_batch(list(texts)), dtype=np.float64)
        return np.asarray([self.score(t) for t in texts], dtype=np.float64)

    def predict(self, text: str) -> bool:
        """Whether the score reaches the model's decision threshold."""
        return self.score(text) >= self.threshold

    def explain(self, text: str, top_k: int = 8) -> list[Contribution]:
        """The n-grams that contribute most to the score, by absolute weight."""
        idx, val = self._features(text)
        contrib = {i: float(self._w64[i]) * float(v) for i, v in zip(idx, val, strict=True)}
        best: dict[int, tuple[str, str]] = {}
        for kind, gram in ngrams(normalize(text, backend=self._impl), self.cfg):
            b = ngram_bucket(kind, gram, self.cfg.dim)
            prev = best.get(b)
            if prev is None or len(gram) > len(prev[1]):
                best[b] = (kind, gram)
        ranked = sorted(contrib, key=lambda b: -abs(contrib[b]))[:top_k]
        return [Contribution(best[b][0], best[b][1], contrib[b]) for b in ranked if b in best]

    # ------------------------------------------------------------- persistence
    def with_threshold(self, threshold: float) -> LinearDetector:
        return LinearDetector(
            self.cfg, self.weights, self.idf, self.bias, threshold, dict(self.meta), self.backend
        )

    def quantized(self) -> LinearDetector:
        """A copy whose weights and IDF values are rounded to float16 (half the file size)."""
        return LinearDetector(
            self.cfg,
            self.weights.astype(np.float16),
            self.idf.astype(np.float16),
            self.bias,
            self.threshold,
            {**self.meta, "storage": "float16"},
            self.backend,
        )

    def save(self, path: str | Path) -> None:
        """Write the model; float16-exact models (see ``quantized``) are stored as float16."""
        half = self.meta.get("storage") == "float16"
        header = {
            "format": 1,
            "features": self.cfg.as_dict(),
            "bias": self.bias,
            "threshold": self.threshold,
            "meta": self.meta,
        }
        with Path(path).open("wb") as fh:
            np.savez_compressed(
                fh,
                weights=self.weights.astype(np.float16) if half else self.weights,
                idf=self.idf.astype(np.float16) if half else self.idf,
                header=np.frombuffer(json.dumps(header).encode(), dtype=np.uint8),
            )

    @classmethod
    def load(cls, path: str | Path, backend: Backend = "auto") -> LinearDetector:
        with np.load(Path(path)) as z:
            header = json.loads(z["header"].tobytes().decode())
            if header.get("format") != 1:
                raise ValueError(f"unsupported model format {header.get('format')!r}")
            return cls(
                FeatureConfig(**header["features"]),
                z["weights"],
                z["idf"],
                float(header["bias"]),
                float(header["threshold"]),
                header.get("meta", {}),
                backend,
            )

    @classmethod
    def load_default(cls, backend: Backend = "auto") -> LinearDetector:
        """The bundled model trained on the full training split (see the model card)."""
        ref = resources.files("prompt_injection_detector") / "assets" / DEFAULT_MODEL
        with resources.as_file(ref) as p:
            return cls.load(p, backend)
