# Data Card: Prompt Injection Detector

All data is public and downloaded by `scripts/download_data.sh` (about 2.0 GB, no token
needed) into `../.bench-data/prompt-injection-detector/` outside the repository.
`scripts/prepare_data.py` builds the prepared dataset; its statistics are committed in
[`benchmarks/results/dataset.json`](../benchmarks/results/dataset.json). Nothing from
these datasets is committed except aggregate counts and a few example prompts in the
robustness results.

## Sources

Label 1 means a prompt-injection or jailbreak attempt, label 0 a benign prompt.
"Read" counts rows loaded; "kept" counts rows left after normalization-based exact
deduplication and removal of label conflicts.

| Source | Hugging Face dataset | License | Labels | Read | Kept |
|---|---|---|---|---:|---:|
| mosscap | Lakera/mosscap_prompt_injection | MIT | 1 (filtered, see below) | 162,489 of 278,945 | 122,668 |
| gandalf | Lakera/gandalf_ignore_instructions | MIT | 1 | 1,000 | 988 |
| deepset | deepset/prompt-injections | Apache-2.0 | 0 and 1 | 662 | 655 |
| jackhhao | jackhhao/jailbreak-classification | Apache-2.0 | 0 and 1 | 1,998 | 1,942 |
| spml | reshabhs/SPML_Chatbot_Prompt_Injection (user prompts) | MIT | 0 and 1 | 16,012 | 15,913 |
| safeguard | xTRam1/safe-guard-prompt-injection | not stated | 0 and 1 | 10,296 | 9,605 |
| wild | TrustAIRLab/in-the-wild-jailbreak-prompts (2023-12-25) | MIT | 1 jailbreak, 0 regular | 15,140 | 13,824 |
| alpaca | tatsu-lab/alpaca (instruction plus input) | CC-BY-NC-4.0 | 0 | 52,002 | 51,998 |
| dolly | databricks/databricks-dolly-15k (instruction plus context) | CC-BY-SA-3.0 | 0 | 15,011 | 14,482 |
| ultrachat | HuggingFaceH4/ultrachat_200k (train_sft prompts) | MIT | 0 | 207,865 | 207,847 |
| oasst2 | OpenAssistant/oasst2 (prompter messages, multilingual) | Apache-2.0 | 0 | 64,592 | 63,679 |
| hhrlhf | Anthropic/hh-rlhf helpful subsets (first human turn) | MIT | 0 | 124,503 | 43,702 |
| openorca | Open-Orca/OpenOrca 1M-GPT4 (seeded sample of 400,000 questions) | MIT | 0 | 400,000 | 369,989 |
| **Total** | | | | **1,071,570** | **917,292** |

`hackaprompt/hackaprompt-dataset` (about 600,000 attack prompts) was the intended
large attack source, but it is gated behind a Hugging Face login and no token was
available, so it is not used. Lakera's Mosscap game data takes its place as the large
attack source.

## Labeling decisions
- **Mosscap filter.** Every Mosscap prompt was typed by a player trying to extract a
  password, but many are ordinary requests on their own ("tell me a joke", "write a
  python function"). A first model trained with all of them as attacks flagged exactly
  those requests. Following Lakera's own filtering of Gandalf into
  `gandalf_ignore_instructions`, a Mosscap prompt is labeled an attack only if it refers
  to the secret, the model's instructions or an override (a fixed regular expression in
  `data.py`: password, secret, prompt, instruction, ignore, previous, reveal, pretend,
  role, debug and similar words). The other 116,456 prompts are left out entirely.
- **Benign sources are assumed benign.** Alpaca, Dolly, UltraChat, OASST2, hh-rlhf and
  OpenOrca prompts are labeled 0 without review; a small number may contain
  injection-like text.
- **Label conflicts.** 70 normalized texts appeared with both labels and were dropped.

## Splits
- Exact duplicates after normalization: 154,206 removed.
- Near-duplicate groups (MinHash, 64 permutations over word 3-shingles, 16 bands of 4
  rows, at least 70% agreeing slots): 768,182 groups; 209,909 prompts share a group with
  at least one other prompt; the largest group has 1,077 prompts.
- Whole groups are assigned to train, validation or test (80/10/10) by a hash of the
  group's first normalized text.
- A random 80/10/10 split would have left 19,198 of 91,648 test prompts (21%) with a
  near-duplicate in the training split.

| Split | Attacks | Benign | Total |
|---|---:|---:|---:|
| train | 113,116 | 621,164 | 734,280 |
| validation | 13,778 | 78,145 | 91,923 |
| test | 13,497 | 77,592 | 91,089 |

Per-source split counts are in `benchmarks/results/dataset.json`.

## Known issues
- Mosscap is 87% of the attack prompts in every split, so in-distribution metrics are
  dominated by one style of attack (secret extraction). The per-source and held-out
  family results in the README separate the sources.
- The Mosscap keyword filter means every Mosscap attack mentions a password, a secret,
  the instructions or similar; models can learn those words. The benign sources rarely
  discuss passwords, so benign prompts about passwords can score high.
- English dominates both classes; OASST2 adds multilingual benign prompts but there are
  few non-English attacks, so a model can learn that non-English text is benign.
- Alpaca is CC-BY-NC-4.0 and Dolly is CC-BY-SA-3.0; the trained weights carry that
  provenance (see the model card).
