# ADR-0002: Hashed n-gram features, FTRL-Proximal training and a Rust core

- **Status:** Accepted
- **Date:** 2026-11-04

## Context
A prompt-injection filter sits in front of every model call, so it must add very little
latency and run without a GPU. Attack prompts share recognizable surface patterns
("ignore the above", "you are now", role-play framing) that character and word
n-grams capture well. Training has to scale to roughly a million prompts on an 8 GB
laptop, and the scoring path must be exactly reproducible across implementations.

## Decision
- Features are character 3 to 5 grams over the normalized text plus word unigrams and
  bigrams, hashed with 64-bit FNV-1a into 2^20 buckets, weighted by sublinear TF times
  smoothed IDF and L2-normalized.
- The classifier is logistic regression trained online with FTRL-Proximal
  (McMahan et al., 2013): per-coordinate learning rates, lazy weights, and memory equal
  to two floats per bucket, so the feature matrix is never materialized.
- Normalization, featurization, training, scoring and near-duplicate grouping run in a
  Rust core compiled into the package with PyO3 and maturin. Featurization runs in
  parallel blocks with rayon; the FTRL updates stay sequential in a SplitMix64-seeded
  order, so results do not depend on the thread count.
- Every Rust function has a Python reference implementation. Parity tests (including
  property-based tests over arbitrary Unicode) check identical normalization, identical
  bucket counts, identical permutations and weights equal to within 1e-6.
- The shipped model stores weights and IDF values as float16 (about 3 MB compressed); it
  is evaluated in that form.

## Alternatives considered
- **Vocabulary-based TF-IDF (scikit-learn TfidfVectorizer):** needs the vocabulary in
  memory (tens of millions of character n-grams at this scale) and a full sparse matrix.
- **Batch L-BFGS over a materialized sparse matrix:** about 1 billion nonzeros at this
  scale, beyond the memory budget.
- **Signed (alternating) hashing:** reduces collision bias but made weights harder to
  read in explanations; with 2^20 buckets collisions cost little (2^18 buckets lose only
  0.2 points of validation TPR at 1% FPR).

## Consequences
- Scoring one prompt takes microseconds and training on 770,000 prompts takes about a
  minute and a half on 8 cores.
- The model explains itself: each score decomposes into per-n-gram contributions.
- A linear model over surface features cannot follow meaning: paraphrases and
  translations that avoid the learned n-grams get through (see the robustness results).
