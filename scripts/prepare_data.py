"""Build the prepared dataset from the raw downloads.

Usage:
    uv run --group data python scripts/prepare_data.py [--raw DIR] [--out FILE]

Writes the prepared parquet file next to the raw data and the dataset statistics to
benchmarks/results/dataset.json.
"""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

from prompt_injection_detector.data import assemble, load_raw

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / ".bench-data" / "prompt-injection-detector"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=DATA / "raw")
    ap.add_argument("--out", type=Path, default=DATA / "prepared.parquet")
    ap.add_argument("--openorca-limit", type=int, default=400_000)
    args = ap.parse_args()
    t0 = time.perf_counter()
    rows = load_raw(args.raw, args.openorca_limit)
    t1 = time.perf_counter()
    data, stats = assemble(rows)
    t2 = time.perf_counter()
    data.save(args.out)
    stats.update(
        openorca_limit=args.openorca_limit,
        load_seconds=round(t1 - t0, 1),
        assemble_seconds=round(t2 - t1, 1),
        peak_rss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
    )
    out = ROOT / "benchmarks" / "results" / "dataset.json"
    out.write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k != "by_source_split"}, indent=2))


if __name__ == "__main__":
    main()
