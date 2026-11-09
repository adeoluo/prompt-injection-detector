"""Check that the transformer detectors give the same test scores under the installed
transformers release as in the committed run (results/test_scores.npz, transformers 4.57.6).

Usage:
    uv run --no-sync python benchmarks/parity_transformers.py [--protectai-rows 3000] [--device cpu]

The fine-tuned e5-small detector rescores the whole test split. The published protectai
model takes about 3.5 hours for the whole split on this machine, so it rescores a seeded
random sample of rows. The committed scores were computed on MPS; a run on another device
also measures that device's floating-point differences. For each detector the script reports
the largest and mean absolute score difference, how many decisions flip at the operating
threshold, and the AUROC and TPR at 1% FPR from the old and the new scores on the same rows.
Writes results/transformers5_parity.json.
"""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
from common import MODELS, PREPARED, RESULTS, environment, peak_rss_mb, write_json

from prompt_injection_detector.data import Dataset
from prompt_injection_detector.metrics import roc_auc, tpr_at_fpr
from prompt_injection_detector.transformer import (
    REFERENCE_MODEL,
    TransformerDetector,
    load_tokenizer,
)

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


def rebuilt_tokenizer_changes(model: str | Path, texts: list[str], max_length: int) -> int:
    """Rows whose token ids differ between the checkpoint's tokenizer.json (what the detector
    uses) and the tokenizer transformers 5 rebuilds through AutoTokenizer."""
    from transformers import AutoTokenizer

    a = load_tokenizer(model)(texts, truncation=True, max_length=max_length)["input_ids"]
    b = AutoTokenizer.from_pretrained(str(model))(texts, truncation=True, max_length=max_length)
    return sum(u != v for u, v in zip(a, b["input_ids"], strict=True))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protectai-rows", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default=None, help="mps, cuda or cpu (default: auto)")
    ap.add_argument(
        "--detectors", nargs="+", choices=("transformer", "protectai"), default=None,
        help="rescore only these; the others are kept from the existing results file",
    )  # fmt: skip
    args = ap.parse_args()
    todo = set(args.detectors or ("transformer", "protectai"))
    test = Dataset.load(PREPARED, split="test")
    y = np.asarray(test.label, dtype=np.int8)
    with np.load(RESULTS / "test_scores.npz") as z:
        if not np.array_equal(z["label"], y):
            raise SystemExit("test_scores.npz does not match the prepared test split")
        old = {k: z[k] for k in ("transformer", "protectai")}
    path = RESULTS / "transformers5_parity.json"
    kept = json.loads(path.read_text())["detectors"] if path.exists() and args.detectors else {}
    results: dict[str, object] = {
        "old": f"transformers {OLD_TRANSFORMERS}",
        "new": f"transformers {version('transformers')}",
        "torch": version("torch"),
        "detectors": {k: v for k, v in kept.items() if k not in todo},
    }

    rng = np.random.default_rng(args.seed)
    idx = np.sort(rng.choice(len(y), args.protectai_rows, replace=False))
    sample = [test.text[i] for i in idx]

    tdir = MODELS / "transformer"
    if "transformer" in todo:
        meta = json.loads((tdir / "detector.json").read_text())
        det = TransformerDetector(
            tdir, max_length=meta["max_length"], batch_size=32, device=args.device
        )
        t0 = time.perf_counter()
        new = det.score_batch(test.text)
        row = compare(y, old["transformer"], new, float(meta["threshold"]))
        row["seconds"] = time.perf_counter() - t0
        row["device"] = det.device
        row["sample_rows_tokenized_differently_by_autotokenizer"] = rebuilt_tokenizer_changes(
            tdir, sample, meta["max_length"]
        )
        results["detectors"]["transformer"] = row  # type: ignore[index]
        print("transformer", row, flush=True)
        del det

    if "protectai" in todo:
        ref = TransformerDetector(REFERENCE_MODEL, max_length=512, batch_size=8, device=args.device)
        t0 = time.perf_counter()
        new = ref.score_batch(sample)
        row = compare(y[idx], old["protectai"][idx], new, 0.5)
        row["seconds"] = time.perf_counter() - t0
        row["device"] = ref.device
        row["sample_seed"] = args.seed
        row["sample_rows_tokenized_differently_by_autotokenizer"] = rebuilt_tokenizer_changes(
            REFERENCE_MODEL, sample, 512
        )
        results["detectors"]["protectai"] = row  # type: ignore[index]
        print("protectai", row, flush=True)

    results["peak_rss_mb"] = peak_rss_mb()
    results["environment"] = environment()
    write_json("transformers5_parity.json", results)


if __name__ == "__main__":
    main()
