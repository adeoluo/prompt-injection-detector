"""Latency and throughput of the linear detector (Rust core and Python reference),
the fine-tuned encoders, the published protectai model and Qwen3Guard-Gen-0.6B.

Usage:
    uv run --no-sync python benchmarks/latency.py [--skip-transformers]

Single-text latency is measured per call through the public Python API on test-split
prompts in their natural length mix; batch throughput uses ``score_batch``.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time

import numpy as np
from common import MODELS, PREPARED, environment, write_json

from prompt_injection_detector.data import Dataset
from prompt_injection_detector.metrics import percentiles
from prompt_injection_detector.model import LinearDetector


def per_call(fn, texts: list[str]) -> dict[str, float]:  # type: ignore[no-untyped-def]
    times = []
    for t in texts:
        t0 = time.perf_counter_ns()
        fn(t)
        times.append((time.perf_counter_ns() - t0) / 1e3)
    out = {k: round(v, 2) for k, v in percentiles(times).items()}
    out["mean"] = round(float(np.mean(times)), 2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-transformers", action="store_true")
    args = ap.parse_args()
    test = Dataset.load(PREPARED).where("test")
    rng = random.Random(0)
    sample = rng.sample(test.text, 10_000)
    lengths = [len(t) for t in sample]
    res: dict[str, object] = {
        "unit": "microseconds per text",
        "sample_chars": {
            "median": float(np.median(lengths)),
            "p95": float(np.percentile(lengths, 95)),
        },
        "environment": environment(),
    }
    rust = LinearDetector.load(MODELS / "linear.npz")
    py = LinearDetector.load(MODELS / "linear.npz", backend="python")
    for t in sample[:200]:
        rust.score(t)
    res["linear_rust_single"] = per_call(rust.score, sample)
    res["linear_python_single"] = per_call(py.score, sample[:1000])
    t0 = time.perf_counter()
    rust.score_batch(test.text)
    secs = time.perf_counter() - t0
    res["linear_rust_batch"] = {
        "texts": len(test.text),
        "seconds": round(secs, 2),
        "texts_per_second": round(len(test.text) / secs),
        "threads": os.cpu_count(),
    }
    if not args.skip_transformers:
        from prompt_injection_detector.guard import GuardDetector
        from prompt_injection_detector.transformer import REFERENCE_MODEL, TransformerDetector

        for name, path, max_len in (
            ("transformer", MODELS / "transformer", 128),
            ("granite", MODELS / "transformer-granite-r2", 128),
            ("protectai", REFERENCE_MODEL, 512),
            ("qwen3guard", None, 512),
        ):
            if path is None:
                det: TransformerDetector | GuardDetector = GuardDetector(
                    batch_size=1, max_text_tokens=max_len
                )
            elif isinstance(path, str) or path.exists():
                det = TransformerDetector(path, max_length=max_len, batch_size=1)
            else:
                continue
            for t in sample[:20]:
                det.score(t)
            res[f"{name}_single"] = per_call(det.score, sample[:500])
            res[f"{name}_device"] = det.device
            del det
            gc.collect()
    write_json("latency.json", res)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
