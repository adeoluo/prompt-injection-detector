"""Train the linear detector on the training split, with a small validation sweep.

Usage:
    uv run --no-sync python benchmarks/train_linear.py [--sweep]

With ``--sweep``, each configuration is trained on a seeded sample of the train split
(``--sweep-sample``, default 200,000 prompts), rounded to float16 storage (the form that
ships), and scored on the validation split; the best by validation TPR at 1% FPR (ties
broken by AUROC) is then trained on the full train split. Its decision threshold is set
at 1% FPR on the validation negatives. Writes the model to the data directory and to the package
assets, and the sweep to results/linear_train.json.
"""

from __future__ import annotations

import argparse
import random
import shutil
import time

import numpy as np
from common import MODELS, PREPARED, ROOT, environment, peak_rss_mb, write_json

from prompt_injection_detector.data import Dataset
from prompt_injection_detector.features import FeatureConfig
from prompt_injection_detector.ftrl import FtrlConfig
from prompt_injection_detector.metrics import roc_auc, threshold_at_fpr, tpr_at_fpr
from prompt_injection_detector.model import LinearDetector

DEFAULT = (FeatureConfig(dim=1 << 20), FtrlConfig(alpha=2.0, l2=1e-4, epochs=3))
SWEEP = [
    (FeatureConfig(dim=1 << 18), FtrlConfig(alpha=2.0, l2=1e-4, epochs=3)),
    (FeatureConfig(dim=1 << 20), FtrlConfig(alpha=0.5, l2=1e-4, epochs=3)),
    (FeatureConfig(dim=1 << 20), FtrlConfig(alpha=1.0, l2=1e-4, epochs=3)),
    (FeatureConfig(dim=1 << 20), FtrlConfig(alpha=2.0, l2=1e-4, epochs=3)),
    (FeatureConfig(dim=1 << 20), FtrlConfig(alpha=2.0, l2=1e-4, epochs=5)),
]


def evaluate_run(
    model: LinearDetector,
    fcfg: FeatureConfig,
    tcfg: FtrlConfig,
    secs: float,
    texts: list[str],
    y: np.ndarray,
) -> dict[str, object]:
    s = model.score_batch(texts)
    run = {
        "features": fcfg.as_dict(),
        "ftrl": tcfg.as_dict(),
        "train_seconds": round(secs, 1),
        "epoch_losses": model.meta["epoch_losses"],
        "val_auroc": roc_auc(y, s),
        "val_tpr_at_1pct_fpr": tpr_at_fpr(y, s, 0.01),
        "val_tpr_at_0.1pct_fpr": tpr_at_fpr(y, s, 0.001),
    }
    print(run, flush=True)
    return run


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", action="store_true", help="try every configuration in SWEEP")
    ap.add_argument("--sweep-sample", type=int, default=200_000)
    args = ap.parse_args()
    data = Dataset.load(PREPARED)
    train, val = data.where("train"), data.where("validation")
    del data
    n_pos = sum(train.label)
    balance = (len(train) - n_pos) / n_pos
    yv = np.asarray(val.label)
    runs = []
    configs = [DEFAULT]
    if args.sweep:
        rng = random.Random(0)
        pick = sorted(rng.sample(range(len(train)), min(args.sweep_sample, len(train))))
        sx = [train.text[i] for i in pick]
        sy = [train.label[i] for i in pick]
        s_bal = (len(sy) - sum(sy)) / sum(sy)
        best_key: tuple[float, float] | None = None
        for fcfg, tcfg in SWEEP:
            tcfg = FtrlConfig(**{**tcfg.as_dict(), "pos_weight": s_bal})
            t0 = time.perf_counter()
            model = LinearDetector.train(sx, sy, fcfg, tcfg).quantized()
            run = evaluate_run(model, fcfg, tcfg, time.perf_counter() - t0, val.text, yv)
            run["n_train"] = len(sx)
            runs.append(run)
            key = (run["val_tpr_at_1pct_fpr"], run["val_auroc"])
            if best_key is None or key > best_key:
                best_key, configs = key, [(fcfg, tcfg)]
        del sx, sy
    fcfg, tcfg = configs[0]
    tcfg = FtrlConfig(**{**tcfg.as_dict(), "pos_weight": balance})
    t0 = time.perf_counter()
    model = LinearDetector.train(train.text, train.label, fcfg, tcfg).quantized()
    final = evaluate_run(model, fcfg, tcfg, time.perf_counter() - t0, val.text, yv)
    final["n_train"] = len(train)
    s = model.score_batch(val.text)
    model = model.with_threshold(threshold_at_fpr(s[yv == 0], 0.01))
    model.meta.update(
        trained_on="train split of the prepared dataset",
        threshold_rule="1% FPR on validation negatives",
        n_validation=len(val),
    )
    MODELS.mkdir(parents=True, exist_ok=True)
    model.save(MODELS / "linear.npz")
    asset = ROOT / "src" / "prompt_injection_detector" / "assets" / "linear-v1.npz"
    asset.parent.mkdir(exist_ok=True)
    shutil.copy(MODELS / "linear.npz", asset)
    write_json(
        "linear_train.json",
        {
            "n_train": len(train),
            "n_train_positive": n_pos,
            "pos_weight": balance,
            "sweep_runs": runs,
            "final": final,
            "selected": {"features": model.cfg.as_dict(), "threshold": model.threshold},
            "peak_rss_mb": peak_rss_mb(),
            "environment": environment(),
        },
    )


if __name__ == "__main__":
    main()
