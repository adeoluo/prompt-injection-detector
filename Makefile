.PHONY: help setup bench-setup lint format type test coverage rust-check rust-develop parity data prepare train-linear train-transformer evaluation bench run diagram docker-build docker-up clean all changelog cut-release

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

setup: ## Install dependencies (compiles the Rust core) and pre-commit hooks
	uv sync --dev
	uv run pre-commit install

format: ## Format automatically
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint (no fixes)
	uv run ruff format --check .
	uv run ruff check .

type: ## Type-check
	uv run mypy src

test: ## Run tests
	uv run pytest

coverage: ## Tests with coverage gate
	uv run pytest --cov=src --cov-report=term-missing --cov-fail-under=80

rust-check: ## Rust core: fmt, clippy (with and without Python), unit tests
	cd rust && cargo fmt --check && cargo clippy --all-targets -- -D warnings && cargo clippy --all-targets --features python -- -D warnings && cargo test

rust-develop: ## Rebuild the Rust core into .venv after editing rust/
	uv sync --group rust
	uv run --no-sync maturin develop --uv --release

parity: ## Rust versus Python parity tests
	PID_REQUIRE_RUST=1 uv run --no-sync pytest tests/test_normalize.py tests/test_features.py tests/test_ftrl.py tests/test_model.py tests/test_dedup.py -v

data: ## Download the public datasets (about 1.9 GB, outside the repository)
	scripts/download_data.sh

prepare: ## Deduplicate and split the downloaded data
	uv run --group data python scripts/prepare_data.py

bench-setup: ## Install everything the full-scale benchmarks need (torch, transformers, scikit-learn)
	uv sync --dev --group data --group validation --group robustness --extra transformer

train-linear: ## Train the linear detector (add ARGS=--sweep for the validation sweep)
	cd benchmarks && uv run --no-sync python train_linear.py $(ARGS)

train-transformer: ## Fine-tune the small transformer on MPS
	cd benchmarks && uv run --no-sync python train_transformer.py

evaluation: ## Reproduce the README results (test split, OOD, robustness, latency)
	cd benchmarks && uv run --no-sync python evaluate.py && uv run --no-sync python ood.py && uv run --no-sync python robustness.py && uv run --no-sync python latency.py

run: ## Run the HTTP service
	uv run python -m prompt_injection_detector serve

diagram: ## Render every diagram in docs/architecture to SVG (house theme, white background)
	@for f in docs/architecture/*.mmd; do \
		npx -y @mermaid-js/mermaid-cli -q -i "$$f" -o "$${f%.mmd}.svg" -c mermaid.config.json -b white || exit 1; \
	done

docker-build: ## Build the image
	docker build -t prompt-injection-detector .

docker-up: ## Run via docker compose
	docker compose up --build

clean: ## Remove caches and artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache dist build *.egg-info coverage.xml rust/target
	find . -type d -name __pycache__ -exec rm -rf {} +

all: lint type coverage ## Full local gate (mirror of continuous integration)

changelog: ## Regenerate CHANGELOG.md from Conventional Commits (git-cliff)
	git cliff --output CHANGELOG.md && perl -0pi -e 's/\n+\z/\n/' CHANGELOG.md

cut-release: ## Cut a release locally: version bump, changelog, tag, GitHub release (RELEASE_VERSION=vX.Y.Z optional)
	scripts/release.sh $(RELEASE_VERSION)
