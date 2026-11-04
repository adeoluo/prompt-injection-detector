# ADR-0003: Undo obfuscation before featurizing

- **Status:** Accepted
- **Date:** 2026-11-04

## Context
Cheap character tricks defeat surface classifiers: Cyrillic or Greek lookalike letters,
zero-width characters between letters, fullwidth or styled Unicode letters, accents,
invisible Unicode tag characters that carry hidden ASCII ("ASCII smuggling"), and base64
payloads with an instruction to decode them. A tokenizer-based model sees different
tokens for each variant, and an n-gram model sees different n-grams.

## Decision
Normalize every text before featurizing, in both training and scoring:
1. decode base64 runs of at least 20 characters that decode to printable UTF-8 and
   append the decoded text;
2. apply Unicode NFKD;
3. map tag characters U+E0020 to U+E007E to the ASCII they hide;
4. lowercase and fold a fixed table of homoglyphs (Cyrillic, Greek, Armenian, small
   capitals) to ASCII;
5. drop combining marks, invisible characters and control characters;
6. collapse whitespace.

Whitespace, invisible characters and controls are defined by explicit code point lists
so the Rust and Python implementations agree exactly.

## Alternatives considered
- **Full Unicode confusables (UTS #39 skeletons):** broader, but folds many legitimate
  non-Latin letters and changes with each Unicode release; a small fixed table covers the
  lookalikes used in practice and is reviewable.
- **Train on augmented data instead of normalizing:** needs the augmentations to be
  anticipated and inflates training; normalization handles the whole class at once.

## Consequences
- The character-level tricks in the robustness suite are neutralized for the linear
  model, and passing normalized text to the transformer detectors also helps them.
- Folding Cyrillic and Greek letters changes how Russian or Greek text is featurized;
  stripping accents merges some distinct words. Both are acceptable for detection.
- Base64 decoding is bounded (at most 8 runs) so long encoded blobs cannot blow up work.
