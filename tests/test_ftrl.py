"""FTRL-Proximal: the reference trainer, the seeded order, and Rust parity."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_injection_detector._backend import rust, rust_available
from prompt_injection_detector.features import FeatureConfig
from prompt_injection_detector.ftrl import Ftrl, FtrlConfig, sigmoid, train_reference
from prompt_injection_detector.rng import SplitMix64, mix64, permutation_reference

needs_rust = pytest.mark.skipif(not rust_available(), reason="Rust core not built")
SMALL = FeatureConfig(dim=1 << 12)


def _xy(corpus: list[dict[str, object]]) -> tuple[list[str], list[float]]:
    return [str(r["text"]) for r in corpus], [float(r["label"]) for r in corpus]  # type: ignore[arg-type]


def test_splitmix_reference_sequence() -> None:
    rng = SplitMix64(0)
    assert rng.next_u64() == 0xE220A8397B1DCDAF
    assert rng.next_u64() == 0x6E789E6AA1B965F4
    assert mix64(0) == 0


@settings(max_examples=50, deadline=None)
@given(st.integers(0, 200), st.integers(0, 2**64 - 1))
def test_permutation_is_a_permutation(n: int, seed: int) -> None:
    assert sorted(permutation_reference(n, seed)) == list(range(n))


@needs_rust
@settings(max_examples=50, deadline=None)
@given(st.integers(0, 300), st.integers(0, 2**64 - 1))
def test_rust_permutation_matches(n: int, seed: int) -> None:
    assert list(rust().permutation(n, seed)) == permutation_reference(n, seed)


def test_reference_trainer_learns(corpus: list[dict[str, object]]) -> None:
    x, y = _xy(corpus)
    w, idf, b, losses = train_reference(x, y, SMALL, FtrlConfig(epochs=3))
    assert losses[-1] < losses[0] < 0.7
    assert len(w) == len(idf) == SMALL.dim
    assert np.count_nonzero(w) > 0 and isinstance(b, float)


@needs_rust
@pytest.mark.parametrize(
    "tcfg",
    [
        FtrlConfig(epochs=2),
        FtrlConfig(alpha=0.1, beta=0.5, l1=0.01, l2=0.001, epochs=3, seed=99, pos_weight=2.5),
    ],
)
def test_rust_trainer_matches_reference(tcfg: FtrlConfig, corpus: list[dict[str, object]]) -> None:
    x, y = _xy(corpus)
    w_b, idf_b, bias, losses = rust().train_ftrl(
        x, y, *SMALL.args(), tcfg.alpha, tcfg.beta, tcfg.l1, tcfg.l2, tcfg.epochs, tcfg.seed,
        tcfg.pos_weight,
    )  # fmt: skip
    w, idf, b, ref_losses = train_reference(x, y, SMALL, tcfg)
    np.testing.assert_allclose(np.frombuffer(w_b, "<f8"), w, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.frombuffer(idf_b, "<f8"), idf, rtol=1e-15)
    assert bias == pytest.approx(b, abs=1e-9)
    np.testing.assert_allclose(losses, ref_losses, rtol=1e-9)


@needs_rust
def test_rust_trainer_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        rust().train_ftrl([], [], *SMALL.args(), 0.5, 1.0, 0.0, 0.0, 1, 0, 1.0)
    with pytest.raises(ValueError):
        rust().train_ftrl(["a"], [], *SMALL.args(), 0.5, 1.0, 0.0, 0.0, 1, 0, 1.0)


def test_reference_trainer_validation() -> None:
    with pytest.raises(ValueError):
        train_reference([], [], SMALL, FtrlConfig())
    with pytest.raises(ValueError):
        train_reference(["a"], [], SMALL, FtrlConfig())


@pytest.mark.parametrize(
    "kwargs", [{"alpha": 0}, {"l1": -1}, {"epochs": 0}, {"pos_weight": 0}, {"beta": -1}]
)
def test_ftrl_config_validation(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        FtrlConfig(**kwargs)  # type: ignore[arg-type]


def test_weights_stay_zero_inside_the_l1_ball() -> None:
    f = Ftrl(4, FtrlConfig(l1=10.0))
    f.update([0, 1], [0.6, 0.8], 1.0)
    assert f.finish()[0] == [0.0] * 4


def test_single_update_matches_closed_form() -> None:
    cfg = FtrlConfig(alpha=0.5, beta=1.0, l1=0.0, l2=0.0)
    f = Ftrl(2, cfg)
    p = f.update([0], [1.0], 1.0)
    assert p == 0.5
    # g = -0.5 for the feature and the bias; sigma * w is zero on the first step.
    assert f.z[0] == pytest.approx(-0.5) and f.n[0] == pytest.approx(0.25)
    w0 = f.weight(0)
    assert w0 == pytest.approx(0.5 / ((1.0 + 0.5) / 0.5))
    assert sigmoid(0.0) == 0.5
