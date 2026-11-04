"""Fine-tune the small transformer detector on a sample of the training split (MPS).

Usage:
    uv run --no-sync python benchmarks/train_transformer.py \
        [--n-train 80000] [--exclude-family FAMILY] [--out DIR]

The decision threshold is set at 1% FPR on the validation negatives. With
``--exclude-family``, every source of that family is removed from training (used by
the out-of-distribution evaluation).
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import time

import numpy as np
from common import MODELS, PREPARED, environment, peak_rss_mb, write_json

from prompt_injection_detector.data import FAMILY, Dataset
from prompt_injection_detector.metrics import roc_auc, threshold_at_fpr, tpr_at_fpr
from prompt_injection_detector.transformer import FineTuneConfig, TransformerDetector, fine_tune


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-train", type=int, default=80_000)
    ap.add_argument("--n-val", type=int, default=40_000)
    ap.add_argument("--exclude-family")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--out")
    args = ap.parse_args()
    rng = random.Random(0)

    def sample(split: str, n: int) -> tuple[list[str], list[int]]:
        part = Dataset.load(PREPARED, split=split)
        keep = [FAMILY[s] != args.exclude_family for s in part.source]
        part = part.select(keep)
        idx = sorted(rng.sample(range(len(part)), min(n, len(part))))
        return [part.text[i] for i in idx], [part.label[i] for i in idx]

    x, y = sample("train", args.n_train)
    vx, vy_list = sample("validation", args.n_val)
    vy = np.asarray(vy_list)
    gc.collect()
    name = args.out or (
        "transformer" + (f"-no-{args.exclude_family}" if args.exclude_family else "")
    )
    out = MODELS / name
    cfg = FineTuneConfig(epochs=args.epochs)
    info = fine_tune(x, y, out, cfg, log=lambda m: print(m, flush=True))
    det = TransformerDetector(out, max_length=cfg.max_length, batch_size=64)
    t0 = time.perf_counter()
    s = det.score_batch(vx)
    info["val_seconds"] = time.perf_counter() - t0
    thr = threshold_at_fpr(s[vy == 0], 0.01)
    info.update(
        n_validation=len(vx),
        val_auroc=roc_auc(vy, s),
        val_tpr_at_1pct_fpr=tpr_at_fpr(vy, s, 0.01),
        threshold=thr,
        exclude_family=args.exclude_family,
        peak_rss_mb=peak_rss_mb(),
        environment=environment(),
    )
    (out / "detector.json").write_text(json.dumps({"threshold": thr, "max_length": cfg.max_length}))
    write_json(f"{name}_train.json", info)
    print({k: v for k, v in info.items() if k != "loss_history"})


if __name__ == "__main__":
    main()
