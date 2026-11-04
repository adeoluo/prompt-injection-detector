# ADR-0005: A small multilingual transformer, and the reference model on the same tests

- **Status:** Accepted
- **Date:** 2026-11-04

## Context
A semantic model complements the n-gram model on paraphrased and translated attacks.
It must fine-tune on Apple MPS within about 3 GB of memory, and the comparison with a
widely used published detector must be on identical test rows.

## Decision
- Fine-tune `intfloat/multilingual-e5-small` (the 12-layer, 384-wide multilingual
  MiniLM architecture, 118M parameters of which 96M are the word embeddings) for binary
  classification, with the word embeddings frozen. Freezing keeps the multilingual
  vocabulary intact for languages the training data barely covers and leaves about 21M
  trainable parameters, so optimizer state stays small.
- Train for one epoch on a seeded 80,000-prompt sample of the training split, with
  length-bucketed batches, class-weighted cross entropy, AdamW and linear warmup and
  decay, at a maximum of 128 tokens.
- Score `protectai/deberta-v3-base-prompt-injection-v2` as published (maximum 512
  tokens, its default 0.5 decision threshold) on the same test rows, and also report the
  threshold-free metrics (AUROC, TPR at fixed FPR) where no threshold choice is involved.

## Alternatives considered
- **deberta-v3-small:** English-only vocabulary, and full fine-tuning needs more memory.
- **English MiniLM:** smaller, but has no path to non-English attacks.
- **Fine-tuning the reference model on this data:** would no longer measure the
  published model.

## Consequences
- The reference model's card states it targets English prompt injection and not
  jailbreaks; results are reported per source so this scope is visible, and it was
  trained on jackhhao/jailbreak-classification, so that source is not out of sample for it.
- The transformer is about a thousand times slower per prompt than the linear model and
  needs the optional `transformer` extra.
