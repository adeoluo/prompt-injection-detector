"""Shared paths and helpers for the benchmark scripts."""

from __future__ import annotations

import json
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path
from typing import Any

# Return Arrow's buffers to the system allocator once the parquet file is converted.
os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / ".bench-data" / "prompt-injection-detector"
PREPARED = DATA / "prepared.parquet"
MODELS = DATA / "models"
RESULTS = ROOT / "benchmarks" / "results"


def write_json(name: str, payload: dict[str, Any]) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def environment() -> dict[str, Any]:
    cpu = "unknown"
    if sys.platform == "darwin":
        cpu = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
        ).stdout.strip()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": cpu,
        "cores": os.cpu_count(),
    }


def peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(rss / 2**20 if sys.platform == "darwin" else rss / 1024, 1)
