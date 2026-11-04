# ADR-0004: Leakage-safe splits and out-of-distribution evaluation by source

- **Status:** Accepted
- **Date:** 2026-11-04

## Context
Public injection datasets are full of exact and near-duplicate prompts: game players
resubmit a prompt with one word changed, and datasets copy from each other. A random
split puts near-copies of test prompts in the training data and overstates accuracy.
Separately, a detector that scores well on the sources it was trained on can still fail
on a new source of attacks, which is the deployment situation that matters.

## Decision
- Drop exact duplicates after normalization, and drop every copy of a text that appears
  with both labels.
- Group near-duplicates with MinHash (64 permutations over word 3-shingles) and banded
  LSH (16 bands of 4 rows), joining a text to the first earlier text in its band bucket
  when their signatures agree on at least 70% of slots, with union-find over the joins.
- Assign each group to train, validation or test by a hash of its first text
  (80/10/10), so a template and its rewrites never straddle splits.
- Report out-of-distribution results by holding out each attack source family
  (Lakera games, SPML, safe-guard, deepset, jackhhao, in-the-wild jailbreaks) from
  training entirely, and scoring that family's test rows.

## Alternatives considered
- **Random split:** 23% of random-split test rows would have a near-duplicate in the
  training split on this data (measured by the build).
- **Exact deduplication only:** misses one-word variants, the most common duplicate.
- **All-pairs verification within LSH buckets:** quadratic in the bucket size; the
  largest buckets (popular game prompts) hold thousands of texts.

## Consequences
- In-distribution numbers are lower than a random split would give and closer to what a
  new deployment sees.
- Comparing a candidate only to the first text of each bucket can miss some pairs; a
  missed pair can still be joined through another band, and the remaining misses are
  rare relative to the leakage removed.
- The held-out-family results show how much each detector depends on having seen a
  source's style, which in-distribution numbers hide.
