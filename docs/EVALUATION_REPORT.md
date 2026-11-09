# Evaluation Report: Prompt Injection Detector

> Reproduce everything here with `make bench-setup data prepare train-linear train-transformer evaluation`.
> Print the tables with `python benchmarks/report.py`. Raw results are in
> [benchmarks/results/](../benchmarks/results/).

## Setup
- **Data:** 13 public sources, 1,071,570 raw rows; 917,292 after removing empty rows,
  154,206 exact duplicates and 70 label conflicts (see [DATA_CARD.md](DATA_CARD.md)).
  MinHash LSH groups near-duplicates (768,182 groups) and whole groups go to one split:
  734,280 train, 91,923 validation, 91,089 test. A random split would have left 19,198 of
  91,648 test rows (21%) with a near-duplicate in training.
- **Metrics:** AUROC; TPR at 1% and 0.1% FPR (threshold free, the share of attacks caught
  when only 1% or 0.1% of benign prompts are flagged); per-source TPR and FPR at each
  detector's operating threshold. 95% intervals are 200 bootstrap resamples of test rows.
- **Detectors:** the linear n-gram model (Rust core, FTRL, 3 epochs, seed 13); a
  scikit-learn `HashingVectorizer` + `SGDClassifier` on the same rows and feature family;
  `intfloat/multilingual-e5-small` fine-tuned on 80,000 training prompts (seed 0); and
  `protectai/deberta-v3-base-prompt-injection-v2` as published (not retrained).
- **Hardware:** Apple M1, 8 cores, 8 GB memory; transformers on MPS. No cloud compute.
- **Runtime:** linear training 15.5 minutes, transformer fine-tuning 39 minutes, scoring
  the test split with the reference model 3.6 hours.
- **Seeds:** one training run per model; intervals reflect test sampling only.
- **Library versions:** the test scores were computed with transformers 4.57.6 and torch 2.15.
  `benchmarks/parity_transformers.py` rescored the test split under transformers 5.22.2
  ([`transformers5_parity.json`](../benchmarks/results/transformers5_parity.json)): the
  fine-tuned e5-small gives identical scores on all 91,089 rows and protectai identical scores
  on a seeded sample of 3,000 rows (the whole split takes 3.5 hours here), so every table below
  holds for 5.22.2. Under transformers 5, `AutoTokenizer` rebuilds the DeBERTa-v3 tokenizer
  without the sentencepiece NFKC normalization stored in its `tokenizer.json`, which changed
  the tokens of 56 of those 3,000 prompts (an ellipsis, fullwidth punctuation or superscript
  digits became `[UNK]`); the detectors load `tokenizer.json` directly, which restores the
  published tokenization. The e5-small tokenizer is not affected (0 of 3,000).

## Headline results

Test split: 91,089 prompts, 13,497 attacks.

| Detector | AUROC (95% CI) | TPR at 1% FPR (95% CI) | TPR at 0.1% FPR | Injection sources only: TPR at 1% FPR | Latency, one prompt (p50) |
|---|---|---|---:|---:|---:|
| **linear n-gram (this repo, Rust)** | 0.9996 (0.9995 to 0.9997) | 99.5% (99.4% to 99.6%) | 96.9% | 99.8% | 27 us |
| scikit-learn hashing + SGD (same data) | 0.9995 (0.9994 to 0.9996) | 99.4% (99.2% to 99.5%) | 94.7% | 99.6% | 464 us |
| **fine-tuned e5-small (this repo)** | 0.9994 (0.9992 to 0.9996) | 99.2% (99.0% to 99.3%) | 95.9% | 99.4% | 14.2 ms |
| protectai deberta-v3-base v2 (published) | 0.9828 (0.9812 to 0.9840) | 87.6% (86.8% to 88.3%) | 67.1% | 87.8% | 60.2 ms |

## Per source

Detection rate on attacks (TPR) and false-positive rate on benign prompts (FPR) at each detector's operating threshold: linear 0.0842, transformer 0.046, protectai 0.5.

| Source | Attacks / benign | **linear n-gram (this repo, Rust)** | **fine-tuned e5-small (this repo)** | protectai deberta-v3-base v2 (published) |
|---|---:|---|---|---|
| alpaca | 0 / 5,253 | FPR 1.18% | FPR 1.01% | FPR 0.19% |
| deepset | 31 / 44 | TPR 87.1%, FPR 9.09% | TPR 48.4%, FPR 4.55% | TPR 32.3%, FPR 2.27% |
| dolly | 0 / 1,452 | FPR 0.48% | FPR 0.41% | FPR 0.00% |
| gandalf | 100 / 0 | TPR 100.0% | TPR 100.0% | TPR 100.0% |
| hhrlhf | 0 / 4,188 | FPR 1.53% | FPR 1.03% | FPR 0.05% |
| jackhhao | 64 / 116 | TPR 87.5%, FPR 9.48% | TPR 92.2%, FPR 26.72% | TPR 90.6%, FPR 0.00% |
| mosscap | 11,741 / 0 | TPR 99.9% | TPR 99.6% | TPR 88.7% |
| oasst2 | 0 / 6,376 | FPR 2.57% | FPR 3.15% | FPR 10.35% |
| openorca | 0 / 37,038 | FPR 0.13% | FPR 0.20% | FPR 0.53% |
| safeguard | 262 / 699 | TPR 98.9%, FPR 2.00% | TPR 98.1%, FPR 1.00% | TPR 82.4%, FPR 0.00% |
| spml | 1,226 / 309 | TPR 99.0%, FPR 12.62% | TPR 99.0%, FPR 9.71% | TPR 98.9%, FPR 10.03% |
| ultrachat | 0 / 20,837 | FPR 0.53% | FPR 0.35% | FPR 0.13% |
| wild | 73 / 1,280 | TPR 67.1%, FPR 15.62% | TPR 67.1%, FPR 18.28% | TPR 63.0%, FPR 11.56% |

## Held-out attack families (linear model)

Each family is removed from a 200,000-prompt training sample; the threshold is set at 1% FPR
on the remaining validation prompts and the model is scored on that family's test rows.

| Held-out family | Test attacks / benign | In-distribution TPR | Held-out TPR | Held-out AUROC vs benign pool | Held-out FPR on family's benign |
|---|---:|---:|---:|---:|---:|
| lakera | 11,841 / 0 | 99.6% | 45.3% | 0.9492 | n/a |
| spml | 1,226 / 309 | 98.3% | 42.1% | 0.9155 | 1.94% |
| safeguard | 262 / 699 | 98.5% | 89.3% | 0.9896 | 2.43% |
| deepset | 31 / 44 | 71.0% | 22.6% | 0.9163 | 4.55% |
| jackhhao | 64 / 116 | 85.9% | 89.1% | 0.9956 | 15.52% |
| wild | 73 / 1,280 | 61.6% | 79.5% | 0.9742 | 46.56% |

## Robustness to rewrites

500 attacks and 500 benign test prompts (English, at most 600 characters), each rewritten
every way below. Cells are attacks detected / benign flagged at each detector's threshold.

| Rewrite | linear | transformer | transformer_normalized_input | protectai | protectai_normalized_input |
|---|---:|---:|---:|---:|---:|
| original | 99.6% / 1.2% | 99.6% / 1.4% | 99.6% / 2.2% | 90.6% / 1.0% | 92.4% / 1.4% |
| homoglyph | 99.6% / 1.2% | 85.0% / 68.0% | 99.6% / 2.2% | 79.4% / 59.0% | 92.4% / 1.4% |
| zero_width | 99.6% / 1.2% | 99.6% / 93.6% | 99.6% / 2.2% | 100.0% / 100.0% | 92.4% / 1.4% |
| tag_smuggle | 99.2% / 0.2% | 0.0% / 0.0% | 99.4% / 4.0% | 0.0% / 0.0% | 88.4% / 0.2% |
| base64 | 99.8% / 3.8% | 99.6% / 100.0% | 100.0% / 73.0% | 100.0% / 100.0% | 100.0% / 89.2% |
| leetspeak | 0.6% / 0.0% | 89.0% / 78.6% | 90.4% / 83.4% | 66.2% / 79.0% | 66.2% / 78.2% |
| letter_spacing | 100.0% / 99.4% | 100.0% / 97.6% | 100.0% / 97.8% | 100.0% / 100.0% | 100.0% / 100.0% |
| fullwidth | 99.6% / 1.2% | 99.6% / 1.4% | 99.6% / 2.2% | 90.6% / 1.0% | 92.4% / 1.4% |
| case_noise | 99.6% / 1.2% | 99.6% / 88.0% | 99.6% / 2.2% | 96.2% / 94.4% | 92.4% / 1.4% |
| synonyms | 98.0% / 1.2% | 99.2% / 1.4% | 99.4% / 1.8% | 84.4% / 1.0% | 87.2% / 1.4% |
| benign_sandwich | 22.8% / 0.0% | 0.0% / 0.0% | 0.0% / 0.0% | 22.8% / 0.0% | 31.0% / 0.0% |
| translate_de | 74.6% / 8.2% | 87.2% / 0.6% | 92.8% / 1.6% | 85.8% / 2.2% | 83.0% / 4.6% |
| translate_fr | 64.4% / 0.8% | 82.4% / 0.2% | 89.0% / 1.2% | 95.2% / 7.8% | 98.4% / 21.6% |
| translate_es | 27.4% / 0.0% | 88.2% / 0.2% | 57.6% / 0.8% | 97.8% / 10.6% | 94.2% / 20.2% |
| llm_paraphrase | 94.6% / 5.4% | 94.4% / 3.0% | 96.4% / 5.6% | 81.6% / 2.2% | 85.8% / 3.6% |

## Latency and throughput

Single prompt: per call through the Python API on test prompts (median 226 characters, p95 2,575). Batch: the whole test split through `score_batch`.

| Detector | Device | p50 | p95 | p99 | Batch throughput |
|---|---|---:|---:|---:|---:|
| Linear n-gram, Rust core | CPU, 1 core | 27 us | 294 us | 501 us | 53,442 texts/s (8 threads) |
| Linear n-gram, Python reference | CPU, 1 core | 1.3 ms | 15.3 ms | 26.8 ms | n/a |
| Fine-tuned e5-small, 128 tokens | MPS | 14.2 ms | 20.3 ms | 23.9 ms | 240 texts/s |
| protectai deberta-v3-base v2, 512 tokens | MPS | 60.2 ms | 482.0 ms | 522.4 ms | 7 texts/s |
| scikit-learn hashing + SGD | CPU | 464 us | n/a | n/a | 2,270 texts/s |

## Error analysis
- **Unseen attack styles:** held-out Lakera game prompts (45.3% caught), SPML (42.1%) and
  deepset (22.6%) show that in-distribution recall overstates recall on new styles.
- **Role-play and persona prompts:** 10% to 18% of benign in-the-wild and SPML prompts are
  flagged by every detector; these are close to jailbreaks in wording.
- **Secret and system-prompt vocabulary:** benign prompts about passwords or system prompts
  score high because most training attacks (Mosscap) use that vocabulary.
- **Rewrites:** leetspeak defeats the n-gram model (0.6% caught); letter spacing makes all
  detectors flag nearly all benign prompts; attacks buried in a benign request are caught at
  most 31% of the time; translation to Spanish lowers n-gram recall to 27.4%.

## Reproducibility
- Data download and preparation: `make data prepare` (no token needed); the dataset summary
  is [dataset.json](../benchmarks/results/dataset.json).
- Training configurations and validation metrics: [linear_train.json](../benchmarks/results/linear_train.json),
  [transformer_train.json](../benchmarks/results/transformer_train.json).
- Every per-row test score: [test_scores.npz](../benchmarks/results/test_scores.npz).
