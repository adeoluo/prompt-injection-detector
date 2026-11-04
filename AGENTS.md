# AGENTS.md

Working notes for coding agents and contributors in **Prompt Injection Detector**. The [README](README.md)
explains what the project does and why; this file explains how to change it safely.

## Project

A prompt-injection and jailbreak classifier with a Rust n-gram core, a fine-tuned small transformer and a leakage-safe benchmark.

## Layout

- `benchmarks/`: benchmark scripts and recorded results
- `docker-compose.yml`: local container setup
- `docs/`: documentation, architecture diagram source and render, design records
- `rust/`: Rust core compiled into a Python extension with PyO3
- `scripts/`: maintenance scripts (release, assets, data)
- `src/`: library and service source
- `tests/`: test suite

Key modules:

- `rust/src/dedup.rs`: Near-duplicate grouping with MinHash and banded LSH, used to build leakage-safe splits: texts in one group always land in the same split.
- `rust/src/features.rs`: Hashed character and word n-gram features with sublinear TF-IDF weighting.
- `rust/src/lib.rs`: Rust core of prompt-injection-detector.
- `rust/src/model.rs`: Linear scoring and FTRL-Proximal training over hashed TF-IDF features.
- `rust/src/normalize.rs`: Text normalization that undoes common obfuscation before featurization.
- `rust/src/python.rs`: PyO3 bindings, built by maturin into the `prompt_injection_detector.pid_rs` module.
- `rust/src/rng.rs`: SplitMix64, the seeded generator shared with the Python reference implementation.
- `src/prompt_injection_detector/__init__.py`: Prompt-injection and jailbreak detection with a Rust n-gram core.
- `src/prompt_injection_detector/attacks.py`: Rule-based adversarial rewrites of an attack prompt, for robustness evaluation.
- `src/prompt_injection_detector/cli.py`: Command line interface: ``python -m prompt_injection_detector <command>``.
- `src/prompt_injection_detector/data.py`: Dataset assembly: 13 public injection, jailbreak and benign prompt sources, exact and near-duplicate removal, and leakage-safe splits.
- `src/prompt_injection_detector/dedup.py`: Near-duplicate grouping with MinHash and banded LSH for leakage-safe splits.
- `src/prompt_injection_detector/features.py`: Hashed character and word n-gram features with sublinear TF-IDF weighting.
- `src/prompt_injection_detector/ftrl.py`: FTRL-Proximal logistic regression (McMahan et al., 2013), reference implementation.
- `src/prompt_injection_detector/metrics.py`: Detection metrics: AUROC, ROC points, TPR at a fixed FPR, thresholds and bootstrap CIs.
- `src/prompt_injection_detector/model.py`: The linear detector: hashed n-gram TF-IDF features scored by a logistic model.
- `src/prompt_injection_detector/normalize.py`: Text normalization that undoes common obfuscation before featurization.
- `src/prompt_injection_detector/rng.py`: SplitMix64, the seeded generator shared with the Rust core.
- `src/prompt_injection_detector/service.py`: HTTP service: scores prompts with the linear detector.
- `src/prompt_injection_detector/transformer.py`: Transformer detectors: fine-tuning a small encoder and scoring with any sequence-classification checkpoint (including the reference model).

## Commands

| Command | Purpose |
|---|---|
| `make help` | Show this help |
| `make setup` | Install dependencies (compiles the Rust core) and pre-commit hooks |
| `make format` | Format automatically |
| `make lint` | Lint (no fixes) |
| `make type` | Type-check |
| `make test` | Run tests |
| `make coverage` | Tests with coverage gate |
| `make rust-check` | Rust core: fmt, clippy (with and without Python), unit tests |
| `make rust-develop` | Rebuild the Rust core into .venv after editing rust/ |
| `make parity` | Rust versus Python parity tests |
| `make data` | Download the public datasets (about 1.9 GB, outside the repository) |
| `make prepare` | Deduplicate and split the downloaded data |
| `make bench-setup` | Install everything the full-scale benchmarks need (torch, transformers, scikit-learn) |
| `make train-linear` | Train the linear detector (add ARGS=--sweep for the validation sweep) |
| `make train-transformer` | Fine-tune the small transformer on MPS |
| `make evaluation` | Reproduce the README results (test split, OOD, robustness, latency) |
| `make run` | Run the HTTP service |
| `make diagram` | Render the architecture diagram to SVG |
| `make docker-build` | Build the image |
| `make docker-up` | Run via docker compose |
| `make clean` | Remove caches and artifacts |
| `make all` | Full local gate (mirror of continuous integration) |
| `make changelog` | Regenerate CHANGELOG.md from Conventional Commits (git-cliff) |
| `make cut-release` | Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional) |

Run the full local gate (`make all`) before committing; CI runs the same checks.

## Conventions

- Commits follow Conventional Commits (`feat:`, `fix:`, `perf:`, `refactor:`, `docs:`, `test:`, `ci:`, `chore:`).
  The changelog and the next version number are derived from them.
- Every number quoted in the README is produced by a command in this repository. When behavior
  changes, rerun that command and update the number in the same commit.
- When a component is added, removed or renamed, update `docs/architecture/architecture.mmd`
  and re-render `architecture.svg` so the diagram matches the code.
- README sections keep their fixed order. Prose avoids em and en dashes and promotional wording.
- Tests accompany behavior changes; a bug fix comes with a test that fails without it.
- Releases are cut with `make cut-release`.

## Python notes

- Python is managed with `uv`; run commands through `uv run` so the locked environment is used.
- Lint and format with `ruff`; type-check with `mypy`; tests use `pytest`.
- Public functions carry type hints and a docstring that states the contract, not the steps.

## Rust notes

- `cargo fmt --check` and `cargo clippy --all-targets -- -D warnings` must pass.
- No `unwrap()` in library code paths that can fail at runtime; return a typed error.
- Unsafe code is limited to the places that need a syscall and carries a `SAFETY:` comment.
