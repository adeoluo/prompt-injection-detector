"""Loads the compiled Rust core and decides which implementation runs.

Every public function takes ``backend="auto" | "rust" | "python"``. ``auto`` uses the
Rust core when it is importable and the Python reference otherwise. Setting the
environment variable ``PID_REQUIRE_RUST=1`` turns a missing extension into an error,
which CI uses so a broken build cannot silently fall back to the slow path.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType
from typing import Literal

Backend = Literal["auto", "rust", "python"]

_rs: ModuleType | None
try:
    _rs = importlib.import_module("prompt_injection_detector.pid_rs")
except ImportError:  # pragma: no cover - exercised only without a compiled build
    _rs = None


def rust_available() -> bool:
    """Whether the compiled Rust core is importable."""
    return _rs is not None


def resolve(backend: Backend) -> Literal["rust", "python"]:
    """The concrete implementation for ``backend``."""
    if backend not in ("auto", "rust", "python"):
        raise ValueError(f"unknown backend {backend!r}")
    if backend == "python":
        return "python"
    if _rs is None:
        if backend == "rust" or os.environ.get("PID_REQUIRE_RUST") == "1":
            raise RuntimeError(
                "the Rust core is not built; reinstall with a Rust toolchain (uv sync)"
            )
        return "python"  # pragma: no cover
    return "rust"


def rust() -> ModuleType:
    """The Rust extension module (raises when it is missing)."""
    if _rs is None:  # pragma: no cover
        raise RuntimeError("the Rust core is not built")
    return _rs
