"""Score the test split with every detector and write the headline comparison.

Usage:
    uv run --no-sync python benchmarks/evaluate.py [--only linear sklearn transformer ...]

Each detector's scores are saved to results/test_scores.npz as soon as it finishes, so a
heavy detector can be rerun on its own (``--only protectai``) in a separate process.

Detectors:
- ``linear``: this package's hashed n-gram model (Rust core), the bundled float16 model;
- ``sklearn``: the same features family in scikit-learn (HashingVectorizer over
  character 3-5 grams plus word 1-2 grams, out-of-core SGD logistic regression,
  3 passes), trained on the same training split;
- ``transformer``: the fine-tuned multilingual-e5-small (MiniLM architecture);
- ``granite``: the fine-tuned granite-embedding-97m-multilingual-r2 (ModernBERT
  architecture), trained the same way;
- ``protectai``: protectai/deberta-v3-base-prompt-injection-v2, used as published;
- ``qwen3guard``: Qwen/Qwen3Guard-Gen-0.6B, used as published; the score is
  P(Unsafe) + P(Controversial) for ranking metrics and the model's own strict label
  (most likely label is Unsafe or Controversial) for the operating point.

Writes results/test_metrics.json, results/test_timing.json and the raw per-row scores
to results/test_scores.npz.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Callable

import numpy as np
import numpy.typing as npt
from common import DATA, MODELS, PREPARED, RESULTS, environment, peak_rss_mb, write_json

from prompt_injection_detector.data import Dataset
from prompt_injection_detector.metrics import (
    bootstrap_ci,
    rates_at,
    roc_auc,
    tpr_at_fpr,
)
from prompt_injection_detector.model import LinearDetector

Scores = npt.NDArray[np.float64]


def sklearn_scores(train: Dataset, test_text: list[str]) -> tuple[Scores, Scores, dict[str, float]]:
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import FeatureUnion

    vec = FeatureUnion(
        [
            ("char", HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5), n_features=1 << 20,
                                       alternate_sign=False, norm="l2")),
            ("word", HashingVectorizer(ngram_range=(1, 2), n_features=1 << 20,
                                       alternate_sign=False, norm="l2")),
        ]
    )  # fmt: skip
    y = np.asarray(train.label)
    pos_w = (len(y) - y.sum()) / y.sum()
    clf = SGDClassifier(loss="log_loss", alpha=1e-6, random_state=0)
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    for _ in range(3):
        order = rng.permutation(len(y))
        for s in range(0, len(order), 20_000):
            idx = order[s : s + 20_000]
            xb = vec.transform([train.text[i] for i in idx])
            yb = y[idx]
            clf.partial_fit(xb, yb, classes=[0, 1], sample_weight=np.where(yb == 1, pos_w, 1.0))
    train_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    scores = np.concatenate(
        [
            clf.predict_proba(vec.transform(test_text[s : s + 20_000]))[:, 1]
            for s in range(0, len(test_text), 20_000)
        ]
    )
    score_s = time.perf_counter() - t0
    one = []
    for t in test_text[:2000]:
        t1 = time.perf_counter()
        clf.predict_proba(vec.transform([t]))
        one.append(time.perf_counter() - t1)
    return scores, np.asarray(one), {"train_seconds": train_s, "score_seconds": score_s}


def timed(fn: Callable[[list[str]], Scores], texts: list[str]) -> tuple[Scores, float]:
    t0 = time.perf_counter()
    s = fn(texts)
    return s, time.perf_counter() - t0


def summarize(
    name: str, y: npt.NDArray[np.int8], s: Scores, thr: float | None
) -> dict[str, object]:
    auc, lo, hi = bootstrap_ci(roc_auc, y, s, n_boot=200, seed=0)
    t1, t1lo, t1hi = bootstrap_ci(lambda a, b: tpr_at_fpr(a, b, 0.01), y, s, n_boot=200, seed=0)
    out: dict[str, object] = {
        "auroc": auc,
        "auroc_ci95": [lo, hi],
        "tpr_at_1pct_fpr": t1,
        "tpr_at_1pct_fpr_ci95": [t1lo, t1hi],
        "tpr_at_0.1pct_fpr": tpr_at_fpr(y, s, 0.001),
    }
    if thr is not None:
        r = rates_at(y, s, thr)
        out["at_threshold"] = {"threshold": thr, "tpr": r.tpr, "fpr": r.fpr,
                               "precision": r.precision, "f1": r.f1}  # fmt: skip
    print(name, {k: v for k, v in out.items() if "ci95" not in k}, flush=True)
    return out


def per_source(
    y: npt.NDArray[np.int8],
    s: Scores,
    src: npt.NDArray[np.str_],
    thr: float,
    decided: Scores | None = None,
) -> dict[str, object]:
    """Per attack source: AUROC of ``s`` against all benign test rows and TPR of
    ``decided`` (default ``s``) at ``thr``; per benign source: FPR at ``thr``."""
    d = s if decided is None else decided
    out: dict[str, object] = {}
    neg = y == 0
    for name in sorted(set(src.tolist())):
        m = src == name
        row: dict[str, float] = {"n_pos": int((m & ~neg).sum()), "n_neg": int((m & neg).sum())}
        if row["n_pos"]:
            sel = (m & ~neg) | neg
            row["auroc_vs_all_benign"] = roc_auc(y[sel], s[sel])
            row["tpr"] = float((d[m & ~neg] >= thr).mean())
        if row["n_neg"]:
            row["fpr"] = float((d[m & neg] >= thr).mean())
        out[name] = row
    return out


DETECTORS = ("linear", "sklearn", "transformer", "granite", "protectai", "qwen3guard")
FINE_TUNED = {"transformer": "transformer", "granite": "transformer-granite-r2"}
SCORES = RESULTS / "test_scores.npz"
TIMING = RESULTS / "test_timing.json"


def _progress(name: str) -> Callable[[int, int], None]:
    return lambda done, total: print(f"{name}: {done:,}/{total:,}", flush=True)


GUARD_CHUNK = 4096
GUARD_CACHE = DATA / "qwen3guard_test"


def guard_probs(test: Dataset) -> tuple[npt.NDArray[np.float64], dict[str, object]]:
    """Qwen3Guard label probabilities for the test split, in resumable chunks of rows.

    Each chunk's probabilities and scoring time are saved under ``GUARD_CACHE``; a rerun
    skips finished chunks, so the run can be stopped and continued.
    """
    from prompt_injection_detector.guard import GUARD_MODEL, GuardDetector

    n = len(test)
    GUARD_CACHE.mkdir(parents=True, exist_ok=True)
    probs = np.empty((n, 3), dtype=np.float64)
    secs = 0.0
    guard: GuardDetector | None = None
    for c, start in enumerate(range(0, n, GUARD_CHUNK)):
        stop = min(start + GUARD_CHUNK, n)
        path = GUARD_CACHE / f"chunk_{c:03d}_{start}_{stop}.npz"
        if not path.exists():
            guard = guard or GuardDetector()
            t0 = time.perf_counter()
            p = guard.label_probs(test.text[start:stop])
            np.savez(path, probs=p, seconds=time.perf_counter() - t0)
        with np.load(path) as z:
            probs[start:stop] = z["probs"]
            secs += float(z["seconds"])
        print(f"qwen3guard: {stop:,}/{n:,} ({secs / 60:.1f} min of scoring)", flush=True)
    info: dict[str, object] = {
        "score_seconds": secs,
        "texts_per_second": n / secs,
        "model": GUARD_MODEL,
        "chunk_rows": GUARD_CHUNK,
    }
    if guard is not None:
        info.update(guard.info())
    return probs, info


def run_detector(name: str, test: Dataset) -> tuple[Scores, dict[str, float]]:
    """Score the test split with one detector; returns scores and timing."""
    n = len(test)
    if name == "linear":
        lin = LinearDetector.load(MODELS / "linear.npz")
        s, secs = timed(lin.score_batch, test.text)
        return s, {"score_seconds": secs, "texts_per_second": n / secs}
    if name == "sklearn":
        train = Dataset.load(PREPARED, split="train")
        s, one, info = sklearn_scores(train, test.text)
        info["texts_per_second"] = n / info["score_seconds"]
        info["single_text_p50_ms"] = float(np.median(one) * 1e3)
        return s, info
    if name == "qwen3guard":
        probs, info = guard_probs(test)
        np.save(RESULTS / "test_qwen3guard_probs.npy", probs.astype(np.float32))
        return probs[:, 1] + probs[:, 2], info  # type: ignore[return-value]
    from prompt_injection_detector.transformer import REFERENCE_MODEL, TransformerDetector

    if name in FINE_TUNED:
        path = MODELS / FINE_TUNED[name]
        meta = json.loads((path / "detector.json").read_text())
        det = TransformerDetector(path, max_length=meta["max_length"], batch_size=32)
    else:
        # Batch 8 at 512 tokens keeps DeBERTa's attention buffers within about 2 GB on MPS.
        det = TransformerDetector(REFERENCE_MODEL, max_length=512, batch_size=8)
    t0 = time.perf_counter()
    s = det.score_batch(test.text, progress=_progress(name))
    secs = time.perf_counter() - t0
    return s, {"score_seconds": secs, "texts_per_second": n / secs, "device": det.device}  # type: ignore[dict-item]


def thresholds_for(names: list[str]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for name in names:
        if name == "linear":
            out[name] = LinearDetector.load(MODELS / "linear.npz").threshold
        elif name in FINE_TUNED:
            meta = json.loads((MODELS / FINE_TUNED[name] / "detector.json").read_text())
            out[name] = float(meta["threshold"])
        elif name in ("protectai", "qwen3guard"):
            out[name] = 0.5  # the published models' own decision rules (see decisions_for)
        else:
            out[name] = None
    return out


def decisions_for(name: str, s: Scores) -> Scores:
    """Scores to threshold at the operating point; Qwen3Guard uses its own strict label."""
    path = RESULTS / "test_qwen3guard_probs.npy"
    if name == "qwen3guard" and path.exists():
        probs = np.load(path).astype(np.float64)
        return (probs.argmax(axis=1) != 0).astype(np.float64)
    return s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--only",
        nargs="+",
        choices=DETECTORS,
        default=list(DETECTORS),
        help="detectors to (re)score; others are read from test_scores.npz",
    )
    args = ap.parse_args()
    test = Dataset.load(PREPARED, split="test")
    y = np.asarray(test.label, dtype=np.int8)
    src = np.asarray(test.source)
    scores: dict[str, Scores] = {}
    if SCORES.exists():
        with np.load(SCORES) as z:
            if np.array_equal(z["label"], y):
                scores = {k: z[k].astype(np.float64) for k in z.files if k in DETECTORS}
    timing: dict[str, dict[str, float]] = json.loads(TIMING.read_text()) if TIMING.exists() else {}
    for name in args.only:
        print(f"scoring with {name}", flush=True)
        s, info = run_detector(name, test)
        scores[name], timing[name] = s, info
        # Save after every detector so a later run can resume with --only.
        np.savez_compressed(SCORES, label=y, source=src,
                            **{k: v.astype(np.float32) for k, v in scores.items()})  # fmt: skip
        write_json(TIMING.name, timing)
        gc.collect()

    names = [k for k in DETECTORS if k in scores]
    thresholds = thresholds_for(names)
    results: dict[str, object] = {
        "n_test": len(y),
        "n_test_positive": int(y.sum()),
        "models": {},
        "per_source": {},
        "timing": timing,
    }
    injection_sources = {"mosscap", "gandalf", "deepset", "spml", "safeguard"}
    inj = np.isin(src, sorted(injection_sources)) | (y == 0)
    for name in names:
        s, thr = scores[name], thresholds[name]
        results["models"][name] = summarize(name, y, s, None)  # type: ignore[index]
        if thr is not None:
            r = rates_at(y, decisions_for(name, s), thr)
            results["models"][name]["at_threshold"] = {  # type: ignore[index]
                "threshold": thr, "tpr": r.tpr, "fpr": r.fpr, "precision": r.precision, "f1": r.f1,
            }  # fmt: skip
        results["models"][name]["injection_sources_only"] = {  # type: ignore[index]
            "auroc": roc_auc(y[inj], s[inj]),
            "tpr_at_1pct_fpr": tpr_at_fpr(y[inj], s[inj], 0.01),
        }
        if thr is not None:
            results["per_source"][name] = per_source(  # type: ignore[index]
                y, s, src, thr, decisions_for(name, s)
            )
    results["peak_rss_mb"] = peak_rss_mb()
    results["environment"] = environment()
    write_json("test_metrics.json", results)


if __name__ == "__main__":
    main()
