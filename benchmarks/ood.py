"""Out-of-distribution evaluation: hold out one source family at a time.

Usage:
    uv run --no-sync python benchmarks/ood.py [--transformer]

For each attack family, a linear model is trained on a seeded sample of the training
split with no row of that family, with its threshold at 1% FPR on the remaining
validation negatives. An in-distribution model trained on an equally sized sample of
all training rows is the comparison, so both see the same amount of data and the gap
is the cost of never having seen the family. Both are scored on the family's test rows.
With ``--transformer``, transformers fine-tuned with ``train_transformer.py
--exclude-family F`` are evaluated the same way where present.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import numpy.typing as npt
from common import MODELS, PREPARED, environment, write_json

from prompt_injection_detector.data import FAMILY, Dataset
from prompt_injection_detector.ftrl import FtrlConfig
from prompt_injection_detector.metrics import roc_auc, threshold_at_fpr
from prompt_injection_detector.model import LinearDetector

FAMILIES = ("lakera", "spml", "safeguard", "deepset", "jackhhao", "wild")


def family_metrics(
    y: npt.NDArray[np.int8], s: npt.NDArray[np.float64], pool: npt.NDArray[np.float64], thr: float
) -> dict[str, float]:
    """TPR and FPR at ``thr`` on the family's rows; AUROC of its attacks against the
    benign pool, and within the family when it has benign rows."""
    pos, neg = s[y == 1], s[y == 0]
    out = {
        "n_pos": int(pos.size),
        "n_neg": int(neg.size),
        "tpr": float((pos >= thr).mean()),
        "auroc_vs_benign_pool": roc_auc(
            np.r_[np.ones(pos.size), np.zeros(pool.size)].astype(int), np.r_[pos, pool]
        ),
    }
    if neg.size:
        out["fpr"] = float((neg >= thr).mean())
        out["auroc_within_family"] = roc_auc(y, s)
    return out


def train_on(
    rows: npt.NDArray[np.int64],
    va: npt.NDArray[np.int64],
    text: list[str],
    label: npt.NDArray[np.int8],
    base: LinearDetector,
    n: int,
) -> tuple[LinearDetector, float]:
    """Train on a seeded sample of ``rows``; threshold at 1% FPR on ``va``."""
    pick = np.sort(np.random.default_rng(0).choice(rows, size=min(n, rows.size), replace=False))
    y = label[pick]
    tcfg = FtrlConfig(**{**base.meta["ftrl"], "pos_weight": float((y == 0).sum() / y.sum())})
    det = LinearDetector.train([text[i] for i in pick], y.tolist(), base.cfg, tcfg).quantized()
    sv = det.score_batch([text[i] for i in va])
    return det, threshold_at_fpr(sv[label[va] == 0], 0.01)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=200_000)
    ap.add_argument("--transformer", action="store_true")
    args = ap.parse_args()
    data = Dataset.load(PREPARED)
    base = LinearDetector.load(MODELS / "linear.npz")
    fam = np.asarray([FAMILY[s] for s in data.source])
    split = np.asarray(data.split)
    label = np.asarray(data.label, dtype=np.int8)
    text = data.text
    pool_idx = np.flatnonzero((split == "test") & (fam == "benign"))
    pool_text = [text[i] for i in pool_idx]
    all_tr = np.flatnonzero(split == "train")
    all_va = np.flatnonzero(split == "validation")
    full, full_thr = train_on(all_tr, all_va, text, label, base, args.n_train)
    full_pool = full.score_batch(pool_text)
    results: dict[str, object] = {
        "n_train_per_model": args.n_train,
        "families": {},
        "environment": environment(),
    }
    for f in FAMILIES:
        tr = np.flatnonzero((split == "train") & (fam != f))
        va = np.flatnonzero((split == "validation") & (fam != f))
        te = np.flatnonzero((split == "test") & (fam == f))
        t0 = time.perf_counter()
        held, thr = train_on(tr, va, text, label, base, args.n_train)
        secs = time.perf_counter() - t0
        te_text = [text[i] for i in te]
        row = {
            "train_seconds": round(secs, 1),
            "in_distribution": family_metrics(
                label[te], full.score_batch(te_text), full_pool, full_thr
            ),
            "held_out": family_metrics(
                label[te], held.score_batch(te_text), held.score_batch(pool_text), thr
            ),
        }
        if args.transformer and (MODELS / f"transformer-no-{f}").exists():
            from prompt_injection_detector.transformer import TransformerDetector

            tin = MODELS / "transformer"
            tout = MODELS / f"transformer-no-{f}"
            for key, d in (("transformer_in_distribution", tin), ("transformer_held_out", tout)):
                meta = json.loads((d / "detector.json").read_text())
                det = TransformerDetector(d, max_length=meta["max_length"], batch_size=64)
                row[key] = family_metrics(
                    label[te], det.score_batch(te_text), det.score_batch(pool_text),
                    meta["threshold"],
                )  # fmt: skip
                del det
        results["families"][f] = row  # type: ignore[index]
        print(f, json.dumps(row), flush=True)
    write_json("ood.json", results)


if __name__ == "__main__":
    main()
