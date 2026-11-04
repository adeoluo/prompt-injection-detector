"""Hashed n-gram features: hashing, n-gram counts, TF-IDF and Rust parity."""

from __future__ import annotations

import math
import unicodedata

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from prompt_injection_detector._backend import rust_available
from prompt_injection_detector.features import (
    FeatureConfig,
    bucket_counts,
    bucket_counts_reference,
    fnv1a,
    idf_from_df,
    ngrams,
    tfidf_reference,
    truncate,
    words,
)

SMALL = FeatureConfig(dim=1 << 12)
needs_rust = pytest.mark.skipif(not rust_available(), reason="Rust core not built")
texts = st.text(
    alphabet=st.characters(blacklist_categories=("Cs", "Cn", "Co"), max_codepoint=0x2FFFF).filter(
        lambda c: unicodedata.category(c) != "Cn"
    ),
    max_size=120,
)


def test_fnv1a_reference_values() -> None:
    # FNV-1a 64 test vectors: "" -> offset basis, "a" (prefix byte 0x61, no data).
    assert fnv1a(0, b"") == 0xAF63BD4C8601B7DF
    assert fnv1a(ord("a"), b"") == 0xAF63DC4C8601EC8C
    assert fnv1a(ord("f"), b"oobar") == 0x85944171F73967E8


def test_words_split_on_space_and_ascii_punctuation() -> None:
    assert words("hello, world! it's ok") == ["hello", "world", "it", "s", "ok"]
    assert words("") == []
    assert words("caf\u00e9 \u00bfqu\u00e9?") == ["caf\u00e9", "\u00bfqu\u00e9"]


def test_ngram_counts_follow_the_formula() -> None:
    grams = list(ngrams("ab cd", SMALL))
    kinds = [k for k, _ in grams]
    assert kinds.count("char") == 5 + 4 + 3
    assert kinds.count("word") == 2
    assert kinds.count("bigram") == 1
    assert list(ngrams("", SMALL)) == []
    no_bigrams = FeatureConfig(dim=1 << 12, word_bigrams=False)
    assert all(k != "bigram" for k, _ in ngrams("ab cd ef", no_bigrams))


def test_truncate_keeps_head_and_tail() -> None:
    assert truncate("abcdefghij", 4) == "ab ij"
    assert truncate("abc", 4) == "abc"


@pytest.mark.parametrize(
    "kwargs",
    [{"dim": 1000}, {"dim": 1}, {"char_min": 0}, {"char_min": 4, "char_max": 3}, {"max_chars": 1}],
)
def test_config_validation(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        FeatureConfig(**kwargs)


def test_counts_are_sorted_and_cover_all_ngrams() -> None:
    idx, cnt = bucket_counts_reference("Ignore all previous instructions", SMALL)
    assert idx == sorted(set(idx))
    assert sum(cnt) == len(list(ngrams("ignore all previous instructions", SMALL)))


def test_tfidf_is_unit_norm_and_sublinear() -> None:
    idf = [2.0] * SMALL.dim
    idx, val = tfidf_reference([1, 2], [1, 4], idf)
    assert idx == [1, 2]
    assert math.isclose(sum(v * v for v in val), 1.0)
    assert math.isclose(val[1] / val[0], 1 + math.log(4))
    assert tfidf_reference([], [], idf) == ([], [])


def test_idf_is_smoothed() -> None:
    idf = idf_from_df([0, 9], 9)
    assert math.isclose(idf[0], math.log(10) + 1)
    assert math.isclose(idf[1], 1.0)


@needs_rust
@settings(max_examples=300, deadline=None)
@given(texts, st.integers(1, 3), st.integers(0, 3), st.booleans(), st.integers(2, 64))
def test_rust_counts_match_python(
    text: str, cmin: int, span: int, bigrams: bool, max_chars: int
) -> None:
    cfg = FeatureConfig(
        dim=1 << 10, char_min=cmin, char_max=cmin + span, word_bigrams=bigrams, max_chars=max_chars
    )
    assert bucket_counts(text, cfg, backend="rust") == bucket_counts_reference(text, cfg)


@needs_rust
def test_rust_rejects_bad_config() -> None:
    from prompt_injection_detector._backend import rust

    with pytest.raises(ValueError):
        rust().bucket_counts("x", 1000, 3, 5, True, 4096)
