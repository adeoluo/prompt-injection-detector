# Model Card: Prompt Injection Detector

## Overview
- **What it is:** two binary classifiers that score a piece of text for prompt-injection
  or jailbreak intent. The linear model ships inside the package; the transformer is
  trained by a script and kept outside the repository.
- **Linear model (`linear-v1`, bundled):** sublinear TF-IDF over hashed character 3 to 5
  grams and word 1 to 2 grams (2^20 buckets) after obfuscation-undoing normalization,
  scored by logistic regression trained with FTRL-Proximal in the Rust core (3 epochs,
  alpha 2.0, beta 1.0, L2 1e-4, seed 13). Weights stored as float16 (3.4 MB file).
  Decision threshold 0.0842, chosen at 1% FPR on the validation split.
- **Transformer:** `intfloat/multilingual-e5-small` fine-tuned for one epoch on a seeded
  80,000-prompt sample of the training split with frozen word embeddings, 128 tokens,
  batch 16, learning rate 5e-5, on Apple MPS (39 minutes). Threshold 0.046 at 1% FPR on
  40,000 validation prompts. Reproduce with `make train-transformer`.
- **Version and date:** 0.1.0, November 2026.

## Intended use
- One layer of defense in front of an LLM application: score user messages, retrieved
  documents and tool outputs before they enter a model's context, then block, flag or
  route to review.
- Research and evaluation of injection detectors on leakage-safe splits.

## Out-of-scope use
- As the only control protecting secrets, tools or consequential actions.
- Content moderation (toxicity, harmful requests that are not injection attempts).
- Scoring system prompts written by the application owner: they contain instruction
  language by design and score high.
- Languages other than English for the linear model (few non-English attacks in training).

## Training data
Thirteen public sources, 917,292 prompts after deduplication; see
[DATA_CARD.md](DATA_CARD.md). Training split: 734,280 prompts, 113,116 attacks.

## Performance
Test split of 91,089 prompts (13,497 attacks) whose near-duplicates never appear in
training. Full numbers and comparisons in [EVALUATION_REPORT.md](EVALUATION_REPORT.md).

| Model | AUROC | TPR at 1% FPR | TPR at 0.1% FPR |
|---|---:|---:|---:|
| linear-v1 | 0.9996 | 99.5% | 96.9% |
| fine-tuned e5-small | 0.9994 | 99.2% | 95.9% |

Known disparities:
- In-distribution numbers are dominated by Mosscap (87% of test attacks). With a whole
  attack family held out of training, recall drops sharply for Lakera game prompts and
  SPML (see the out-of-distribution table in the README).
- False-positive rates differ by benign source: about 0.1% on OpenOrca questions, 2.6%
  on multilingual OASST2 messages, 9% to 16% on the benign halves of the small
  role-play and jailbreak datasets (deepset, jackhhao, SPML, in-the-wild).
- Meaning-preserving rewrites (paraphrase and translation) lower recall the most; see the
  robustness table.

## Risks and limitations
- An attacker who can query the detector can search for a rewrite that passes.
- Labels are noisy: the Mosscap filter is a keyword rule, and benign sources were not
  reviewed one by one.
- Benign prompts about passwords, secrets or system prompts score higher than they
  should, because most attack prompts mention them.
- **Dual use:** the attack rewrites in `attacks.py` describe well-known obfuscation
  tricks and are applied only to public dataset prompts; they add no new capability.

## Safety mitigations
- Normalization undoes homoglyphs, zero-width and tag characters, fullwidth letters,
  accents and base64 before scoring (ADR 0003).
- Thresholds are explicit and adjustable per application; the service accepts a
  per-request threshold.
- `explain` shows the n-grams behind a score for review; do not expose it to untrusted
  callers.

## Licensing and provenance
The bundled weights were trained on data that includes Alpaca (CC-BY-NC-4.0) and
Dolly (CC-BY-SA-3.0). Treat the weights as carrying those terms; retrain without those
sources for unrestricted commercial use.

## Environmental and cost footprint
Apple M1 laptop, 8 cores, 8 GB memory. Linear training: 15.5 minutes on CPU. Transformer
fine-tuning: 39 minutes on MPS. Scoring the 91,089-prompt test split with the reference
DeBERTa model took 3.6 hours on MPS. No cloud compute.
