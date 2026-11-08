"""Check that the transformer detectors give the same test scores under the installed
transformers release as in the committed run (results/test_scores.npz, transformers 4.57.6).

Usage:
    uv run --no-sync python benchmarks/parity_transformers.py [--protectai-rows 3000]

The fine-tuned e5-small detector rescores the whole test split. The published protectai
model takes about 3.5 hours for the whole split on this machine, so it rescores a seeded
random sample of rows. For each detector the script reports the largest and mean absolute
score difference, how many decisions flip at the operating threshold, and the AUROC and
TPR at 1% FPR from the old and the new scores on the same rows. Writes
results/transformers5_parity.json.
"""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version

import numpy as np
from common import MODELS, PREPARED, RESULTS, environment, peak_rss_mb, write_json

from prompt_injection_detector.data import Dataset
from prompt_injection_detector.metrics import roc_auc, tpr_at_fpr
from prompt_injection_detector.transformer import REFERENCE_MODEL, TransformerDetector

OLD_TRANSFORMERS = "4.57.6"


def compare(y: np.ndarray, old: np.ndarray, new: np.ndarray, thr: float) -> dict[str, float | int]:
    diff = np.abs(old.astype(np.float64) - new)
    out: dict[str, float | int] = {
        "rows": len(y),
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "decision_flips": int(((old >= thr) != (new >= thr)).sum()),
        "threshold": thr,
    }
    if 0 < y.sum() < len(y):
        out.update(
            auroc_old=roc_auc(y, old.astype(np.float64)),
            auroc_new=roc_auc(y, new),
            tpr_at_1pct_fpr_old=tpr_at_fpr(y, old.astype(np.float64), 0.01),
            tpr_at_1pct_fpr_new=tpr_at_fpr(y, new, 0.01),
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protectai-rows", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    test = Dataset.load(PREPARED, split="test")
    y = np.asarray(test.label, dtype=np.int8)
    with np.load(RESULTS / "test_scores.npz") as z:
        if not np.array_equal(z["label"], y):
            raise SystemExit("test_scores.npz does not match the prepared test split")
        old = {k: z[k] for k in ("transformer", "protectai")}
    results: dict[str, object] = {
        "old": f"transformers {OLD_TRANSFORMERS}",
        "new": f"transformers {version('transformers')}",
        "torch": version("torch"),
        "detectors": {},
    }

    tdir = MODELS / "transformer"
    meta = json.loads((tdir / "detector.json").read_text())
    det = TransformerDetector(tdir, max_length=meta["max_length"], batch_size=32)
    t0 = time.perf_counter()
    new = det.score_batch(test.text)
    row = compare(y, old["transformer"], new, float(meta["threshold"]))
    row["seconds"] = time.perf_counter() - t0
    row["device"] = det.device
    results["detectors"]["transformer"] = row  # type: ignore[index]
    print("transformer", row, flush=True)
    del det

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(y), args.protectai_rows, replace=False))
    ref = TransformerDetector(REFERENCE_MODEL, max_length=512, batch_size=8)
    t0 = time.perf_counter()
    new = ref.score_batch([test.text[i] for i in idx])
    row = compare(y[idx], old["protectai"][idx], new, 0.5)
    row["seconds"] = time.perf_counter() - t0
    row["device"] = ref.device
    row["sample_seed"] = args.seed
    results["detectors"]["protectai"] = row  # type: ignore[index]
    print("protectai", row, flush=True)

    results["peak_rss_mb"] = peak_rss_mb()
    results["environment"] = environment()
    write_json("transformers5_parity.json", results)


if __name__ == "__main__":
    main()
