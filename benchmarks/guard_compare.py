"""Qwen3Guard-Gen-0.6B against the detectors here, on the same sampled test rows and rewrites.

Usage:
    uv run --no-sync python benchmarks/guard_compare.py [--per-class 1000] [--robustness 100]

Qwen3Guard-Gen is a 0.6B generative moderation model; on this shared 8 GB machine it scores
well under one text per second through MPS, so the whole 91,089-row test split (hours for the
other detectors, days for this one) is out of reach. It is compared on a seeded sample of the
test split with the same number of attacks and benign prompts, and every other detector is
scored on exactly those rows from its committed per-row test scores
(``results/test_scores.npz``). With ``--robustness N`` it also scores the first N attacks and
N benign prompts of a set of rewrites from the robustness benchmark, next to the linear model,
the fine-tuned e5-small and the protectai model on the same texts.

Qwen3Guard's ranking score is P(Unsafe) + P(Controversial) after its ``Safety:`` prefix; its
operating point is the model's own strict label (the most likely label is Unsafe or
Controversial). The other detectors use their usual thresholds. Qwen3Guard scores are cached
per chunk under the data directory, so an interrupted run resumes.

Writes results/qwen3guard_compare.json.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import numpy.typing as npt
from common import DATA, MODELS, PREPARED, RESULTS, environment, peak_rss_mb, write_json

from prompt_injection_detector.data import Dataset
from prompt_injection_detector.metrics import bootstrap_ci, rates_at, roc_auc, tpr_at_fpr
from prompt_injection_detector.model import LinearDetector

CACHE = DATA / "qwen3guard"
CHUNK = 250
ROBUSTNESS_VARIANTS = (
    "original",
    "homoglyph",
    "zero_width",
    "base64",
    "leetspeak",
    "translate_de",
    "llm_paraphrase",
)
DETECTORS = ("linear", "sklearn", "transformer", "protectai")


def balanced_sample(y: npt.NDArray[np.int8], per_class: int, seed: int) -> npt.NDArray[np.int64]:
    """Sorted row indices: ``per_class`` attacks and ``per_class`` benign rows, seeded."""
    rng = np.random.default_rng(seed)
    pos = rng.choice(np.flatnonzero(y == 1), per_class, replace=False)
    neg = rng.choice(np.flatnonzero(y == 0), per_class, replace=False)
    return np.sort(np.concatenate([pos, neg]))


class CachedGuard:
    """Qwen3Guard label probabilities, cached in chunks keyed by a name."""

    def __init__(self) -> None:
        self._guard: Any = None
        self.seconds = 0.0
        self.scored = 0

    def guard(self) -> Any:
        if self._guard is None:
            from prompt_injection_detector.guard import GuardDetector

            self._guard = GuardDetector()
        return self._guard

    def probs(self, key: str, texts: list[str]) -> npt.NDArray[np.float64]:
        out = np.empty((len(texts), 3), dtype=np.float64)
        CACHE.mkdir(parents=True, exist_ok=True)
        for c, start in enumerate(range(0, len(texts), CHUNK)):
            stop = min(start + CHUNK, len(texts))
            path = CACHE / f"{key}_{c:03d}_{start}_{stop}.npz"
            if not path.exists():
                t0 = time.perf_counter()
                p = self.guard().label_probs(texts[start:stop])
                np.savez(path, probs=p, seconds=time.perf_counter() - t0)
            with np.load(path) as z:
                out[start:stop] = z["probs"]
                self.seconds += float(z["seconds"])
                self.scored += stop - start
            mins = self.seconds / 60
            print(f"{key}: {stop}/{len(texts)} ({mins:.1f} min of scoring)", flush=True)
        return out


def decide(probs: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    return (probs.argmax(axis=1) != 0).astype(np.float64)


def thresholds() -> dict[str, float]:
    out = {"linear": LinearDetector.load(MODELS / "linear.npz").threshold, "protectai": 0.5}
    meta = json.loads((MODELS / "transformer" / "detector.json").read_text())
    out["transformer"] = float(meta["threshold"])
    return out


def summary(
    y: npt.NDArray[np.int8], s: npt.NDArray[np.float64], decided: npt.NDArray[np.bool_] | None
) -> dict[str, Any]:
    auc, lo, hi = bootstrap_ci(roc_auc, y, s, n_boot=200, seed=0)
    t1, t1lo, t1hi = bootstrap_ci(lambda a, b: tpr_at_fpr(a, b, 0.01), y, s, n_boot=200, seed=0)
    out: dict[str, Any] = {
        "auroc": auc,
        "auroc_ci95": [lo, hi],
        "tpr_at_1pct_fpr": t1,
        "tpr_at_1pct_fpr_ci95": [t1lo, t1hi],
    }
    if decided is not None:
        r = rates_at(y, decided.astype(np.float64), 0.5)
        out["at_operating_point"] = {"tpr": r.tpr, "fpr": r.fpr, "precision": r.precision}
    return out


def test_sample(args: argparse.Namespace, guard: CachedGuard) -> dict[str, Any]:
    test = Dataset.load(PREPARED, split="test")
    y_all = np.asarray(test.label, dtype=np.int8)
    idx = balanced_sample(y_all, args.per_class, args.seed)
    y = y_all[idx]
    src = np.asarray(test.source)[idx]
    texts = [test.text[i] for i in idx]
    probs = guard.probs(f"test_s{args.seed}_n{args.per_class}", texts)
    with np.load(RESULTS / "test_scores.npz") as z:
        if not np.array_equal(z["label"], y_all):
            raise SystemExit("test_scores.npz does not match the prepared test split")
        stored = {k: z[k][idx].astype(np.float64) for k in DETECTORS if k in z.files}
    thr = thresholds()
    models: dict[str, Any] = {}
    for name, s in stored.items():
        decided = s >= thr[name] if name in thr else None
        models[name] = summary(y, s, decided)
    guard_score = probs[:, 1] + probs[:, 2]
    models["qwen3guard"] = summary(y, guard_score, decide(probs).astype(bool))
    decisions = {k: v >= thr[k] for k, v in stored.items() if k in thr}
    decisions["qwen3guard"] = decide(probs).astype(bool)
    per_source: dict[str, Any] = {}
    for name in sorted(set(src.tolist())):
        m = src == name
        pos, neg = m & (y == 1), m & (y == 0)
        row: dict[str, Any] = {"n_pos": int(pos.sum()), "n_neg": int(neg.sum())}
        for det, d in decisions.items():
            cell: dict[str, float] = {}
            if pos.any():
                cell["tpr"] = float(d[pos].mean())
            if neg.any():
                cell["fpr"] = float(d[neg].mean())
            row[det] = cell
        per_source[name] = row
    label_counts = np.bincount(probs.argmax(axis=1), minlength=3)
    return {
        "rows": len(idx),
        "per_class": args.per_class,
        "seed": args.seed,
        "models": models,
        "per_source_at_operating_point": per_source,
        "qwen3guard_label_counts": {
            "attacks": np.bincount(probs[y == 1].argmax(axis=1), minlength=3).tolist(),
            "benign": np.bincount(probs[y == 0].argmax(axis=1), minlength=3).tolist(),
            "all": label_counts.tolist(),
            "labels": ["Safe", "Unsafe", "Controversial"],
        },
    }


def robustness_subset(n: int, guard: CachedGuard) -> dict[str, Any]:
    from prompt_injection_detector.normalize import normalize
    from prompt_injection_detector.transformer import REFERENCE_MODEL, TransformerDetector

    cached = json.loads((DATA / "robustness_texts.json").read_text())
    if cached.get("_meta", {}).get("paraphrase_model") is None:
        raise SystemExit("run robustness.py first so the paraphrases are current")
    thr = thresholds()
    meta = json.loads((MODELS / "transformer" / "detector.json").read_text())
    texts = {v: (cached["attack"][v][:n], cached["benign"][v][:n]) for v in ROBUSTNESS_VARIANTS}
    rows: dict[str, dict[str, Any]] = {v: {} for v in ROBUSTNESS_VARIANTS}

    def run(scorers: dict[str, Any]) -> None:
        for v, (att, ben) in texts.items():
            for name, fn in scorers.items():
                t = thr[name.removesuffix("_normalized_input")]
                rows[v][name] = {
                    "detection_rate": float((np.asarray(fn(att)) >= t).mean()),
                    "false_positive_rate": float((np.asarray(fn(ben)) >= t).mean()),
                }

    # One model in memory at a time: this machine is shared and has 8 GB.
    run({"linear": LinearDetector.load(MODELS / "linear.npz").score_batch})
    e5 = TransformerDetector(MODELS / "transformer", max_length=meta["max_length"], batch_size=32)
    run(
        {
            "transformer": e5.score_batch,
            "transformer_normalized_input": lambda ts, m=e5: m.score_batch(
                [normalize(t) for t in ts]
            ),
        }
    )
    del e5
    release_memory()
    ref = TransformerDetector(REFERENCE_MODEL, max_length=512, batch_size=8)
    run({"protectai": ref.score_batch})
    del ref
    release_memory()
    out: dict[str, Any] = {"n_per_class": n, "variants": {}}
    for v, (att, ben) in texts.items():
        pa = guard.probs(f"robust_{v}_att_n{n}", att)
        pb = guard.probs(f"robust_{v}_ben_n{n}", ben)
        rows[v]["qwen3guard"] = {
            "detection_rate": float(decide(pa).mean()),
            "false_positive_rate": float(decide(pb).mean()),
        }
        out["variants"][v] = rows[v]
        print(v, json.dumps(rows[v]), flush=True)
    return out


def release_memory() -> None:
    import gc

    import torch

    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--per-class", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--robustness", type=int, default=0, help="texts per class and rewrite")
    args = ap.parse_args()
    path = RESULTS / "qwen3guard_compare.json"
    results: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {}
    guard = CachedGuard()
    results["test_sample"] = test_sample(args, guard)
    write_json(path.name, results)
    if args.robustness:
        results["robustness"] = robustness_subset(args.robustness, guard)
    results["model"] = "Qwen/Qwen3Guard-Gen-0.6B"
    results["qwen3guard_scoring"] = {
        "texts": guard.scored,
        "seconds": guard.seconds,
        "texts_per_second": guard.scored / guard.seconds if guard.seconds else None,
        "device": "mps",
        "note": "summed over cached chunks; the machine was shared with other jobs",
    }
    results["peak_rss_mb"] = peak_rss_mb()
    results["environment"] = environment()
    write_json(path.name, results)
    print(json.dumps(results["test_sample"]["models"], indent=1))


if __name__ == "__main__":
    main()
