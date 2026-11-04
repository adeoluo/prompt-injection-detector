#!/usr/bin/env bash
# Download the public datasets used for training and evaluation (no token needed).
#
# Usage: scripts/download_data.sh [DATA_DIR]
# Default DATA_DIR: ../.bench-data/prompt-injection-detector (outside the repository).
# Total download is about 1.9 GB. Files that already exist are skipped.
set -euo pipefail

DATA_DIR="${1:-$(cd "$(dirname "$0")/../.." && pwd)/.bench-data/prompt-injection-detector}"
RAW="$DATA_DIR/raw"
mkdir -p "$RAW"
HF="https://huggingface.co/datasets"

fetch() {  # fetch <repo> <path in repo> <local name>
  local out="$RAW/$3"
  if [[ -s "$out" ]]; then echo "skip $3"; return; fi
  echo "get  $3"
  curl -fsSL --retry 3 -o "$out.part" "$HF/$1/resolve/main/$2" && mv "$out.part" "$out"
}

# Injection and jailbreak sources
fetch Lakera/mosscap_prompt_injection data/train-00000-of-00001-07ae0ed17fa07cc1.parquet mosscap-train.parquet
fetch Lakera/mosscap_prompt_injection data/validation-00000-of-00001-e136a45438d15f16.parquet mosscap-validation.parquet
fetch Lakera/mosscap_prompt_injection data/test-00000-of-00001-06acab2d6a2bcf9f.parquet mosscap-test.parquet
fetch Lakera/gandalf_ignore_instructions data/train-00000-of-00001-ded53be747ff55cd.parquet gandalf-train.parquet
fetch Lakera/gandalf_ignore_instructions data/validation-00000-of-00001-94481a2a09ff2fff.parquet gandalf-validation.parquet
fetch Lakera/gandalf_ignore_instructions data/test-00000-of-00001-bc92128b9288a6d1.parquet gandalf-test.parquet
fetch deepset/prompt-injections data/train-00000-of-00001-9564e8b05b4757ab.parquet deepset-train.parquet
fetch deepset/prompt-injections data/test-00000-of-00001-701d16158af87368.parquet deepset-test.parquet
fetch jackhhao/jailbreak-classification default/jailbreak_dataset_full.csv jackhhao-full.csv
fetch reshabhs/SPML_Chatbot_Prompt_Injection spml_prompt_injection.csv spml.csv
fetch xTRam1/safe-guard-prompt-injection data/train-00000-of-00001.parquet safeguard-train.parquet
fetch xTRam1/safe-guard-prompt-injection data/test-00000-of-00001.parquet safeguard-test.parquet
fetch TrustAIRLab/in-the-wild-jailbreak-prompts jailbreak_2023_12_25/train-00000-of-00001.parquet wild-jailbreak.parquet
fetch TrustAIRLab/in-the-wild-jailbreak-prompts regular_2023_12_25/train-00000-of-00001.parquet wild-regular.parquet

# Benign prompt sources (false-positive measurement at scale)
fetch tatsu-lab/alpaca data/train-00000-of-00001-a09b74b3ef9c3b56.parquet alpaca.parquet
fetch databricks/databricks-dolly-15k databricks-dolly-15k.jsonl dolly.jsonl
fetch HuggingFaceH4/ultrachat_200k data/train_sft-00000-of-00003-a3ecf92756993583.parquet ultrachat-0.parquet
fetch HuggingFaceH4/ultrachat_200k data/train_sft-00001-of-00003-0a1804bcb6ae68c6.parquet ultrachat-1.parquet
fetch HuggingFaceH4/ultrachat_200k data/train_sft-00002-of-00003-ee46ed25cfae92c6.parquet ultrachat-2.parquet
fetch OpenAssistant/oasst2 2023-11-05_oasst2_prompts.messages.jsonl.gz oasst2-prompts.jsonl.gz
fetch Open-Orca/OpenOrca 1M-GPT4-Augmented.parquet openorca-gpt4.parquet

du -sh "$RAW"
