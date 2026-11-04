"""Backend selection."""

from __future__ import annotations

import pytest

from prompt_injection_detector import _backend
from prompt_injection_detector._backend import resolve, rust_available


def test_resolve() -> None:
    assert resolve("python") == "python"
    assert resolve("auto") == ("rust" if rust_available() else "python")
    with pytest.raises(ValueError):
        resolve("gpu")  # type: ignore[arg-type]


def test_missing_rust_is_an_error_when_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_backend, "_rs", None)
    with pytest.raises(RuntimeError):
        resolve("rust")
    monkeypatch.setenv("PID_REQUIRE_RUST", "1")
    with pytest.raises(RuntimeError):
        resolve("auto")
