# Changelog

All notable changes to this project are documented here. The file is generated
from Conventional Commits by [git-cliff](https://git-cliff.org) when a release is cut
with `make cut-release`.

## [0.1.0] 2026-11-04

### Features

- Rust core for normalization, hashed n-gram features, FTRL training and near-duplicate grouping (#1)
- Dataset assembly from 12 public sources with exact and near-duplicate removal and leakage-safe splits (#2)
- Add conversational benign prompts from hh-rlhf and label only Mosscap prompts that target the secret or instructions (#3)
- Linear detector with a bundled model, attack rewrites, metrics, CLI and HTTP service (#5)
- Transformer detector and the benchmark suite (#7)
- Benchmark results, evaluation report and README with demos (#9)

### Bug fixes

- Fold Greek capitals that look like Latin letters, and add behavior tests for the bundled model (#6)

### Documentation

- Replace the template license line in the README (#4)

### Maintenance

- Scaffold project
- Release tooling, changelog and contributor notes (#8)
