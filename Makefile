SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

PYTHON ?= python3
CARGO ?= cargo
MDBOOK ?= mdbook

SITE_OUTPUT_DIR ?= /tmp/robowbc-site
SITE_BIND ?= 127.0.0.1
SITE_PORT ?= 8000
SITE_OPEN ?= 0
MUJOCO_DOWNLOAD_DIR ?= $(CURDIR)/.cache/mujoco
ROBOWBC_BINARY ?= $(CURDIR)/target/debug/robowbc
SITE_PYTHON_DEPS ?= numpy joblib onnxruntime==1.24.4 pyyaml mujoco Pillow

.DEFAULT_GOAL := help

.PHONY: help toolchain build build-release check test clippy fmt fmt-check rust-doc docs-book docs verify smoke models-public site-python-deps benchmark-robowbc benchmark-official benchmark-summary benchmark-nvidia site site-smoke site-serve

help: ## Show available targets and useful variables.
	@awk 'BEGIN {FS = ":.*## "; print "Targets:"} /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nVariables:\n"
	@printf "  %-20s %s\n" "PYTHON" "$(PYTHON)"
	@printf "  %-20s %s\n" "SITE_OUTPUT_DIR" "$(SITE_OUTPUT_DIR)"
	@printf "  %-20s %s\n" "MUJOCO_DOWNLOAD_DIR" "$(MUJOCO_DOWNLOAD_DIR)"
	@printf "  %-20s %s\n" "SITE_PORT" "$(SITE_PORT)"
	@printf "  %-20s %s\n" "SITE_OPEN" "$(SITE_OPEN)"

toolchain: ## Print the Rust toolchain versions used by this repo.
	rustc --version
	$(CARGO) --version

build: ## Build the Rust workspace in debug mode.
	$(CARGO) build

build-release: ## Build the Rust workspace in release mode.
	$(CARGO) build --release

check: ## Run cargo check across the workspace and all targets.
	$(CARGO) check --workspace --all-targets

test: ## Run cargo test across the workspace and all targets.
	$(CARGO) test --workspace --all-targets

clippy: ## Run clippy with warnings promoted to errors.
	$(CARGO) clippy --workspace --all-targets -- -D warnings

fmt: ## Format the Rust workspace.
	$(CARGO) fmt --all

fmt-check: ## Verify Rust formatting without changing files.
	$(CARGO) fmt --all -- --check

rust-doc: ## Build Rust API docs.
	$(CARGO) doc --workspace --no-deps

docs-book: ## Build the mdBook docs (requires mdbook in PATH).
	if ! command -v $(MDBOOK) >/dev/null; then \
		echo "mdbook not found in PATH; install it before running 'make docs-book'." >&2; \
		exit 1; \
	fi
	$(MDBOOK) build

docs: rust-doc docs-book ## Build both Rust API docs and mdBook docs.

verify: check test clippy fmt-check ## Run the standard Rust validation suite.

smoke: ## Run the no-download local decoupled_wbc smoke config.
	$(CARGO) run --bin robowbc -- run --config configs/decoupled_smoke.toml

models-public: ## Download the public policy checkpoints used by the site and benchmarks.
	bash scripts/download_gear_sonic_models.sh models/gear-sonic
	bash scripts/download_decoupled_wbc_models.sh models/decoupled-wbc
	bash scripts/download_wbc_agile_models.sh models/wbc-agile
	bash scripts/download_bfm_zero_models.sh models/bfm_zero

site-python-deps: ## Install Python dependencies required for site generation and proof-pack capture.
	$(PYTHON) -m pip install $(SITE_PYTHON_DEPS)

benchmark-robowbc: ## Regenerate normalized RoboWBC benchmark artifacts.
	$(PYTHON) scripts/bench_robowbc_compare.py --all

benchmark-official: ## Regenerate normalized official NVIDIA benchmark artifacts.
	$(PYTHON) scripts/bench_nvidia_official.py --all

benchmark-summary: ## Render benchmark Markdown and HTML summaries from the normalized artifacts.
	$(PYTHON) scripts/render_nvidia_benchmark_summary.py \
		--root artifacts/benchmarks/nvidia \
		--output artifacts/benchmarks/nvidia/SUMMARY.md \
		--html-output artifacts/benchmarks/nvidia/index.html

benchmark-nvidia: benchmark-robowbc benchmark-official benchmark-summary ## Rebuild the full NVIDIA benchmark comparison package.

site: ## Build the full static site bundle with policy pages, proof packs, and benchmarks.
	MUJOCO_DOWNLOAD_DIR="$(MUJOCO_DOWNLOAD_DIR)" \
	$(PYTHON) scripts/build_site.py \
		--repo-root . \
		--robowbc-binary "$(ROBOWBC_BINARY)" \
		--output-dir "$(SITE_OUTPUT_DIR)"

site-smoke: ## Validate the generated site bundle layout and embedded playback paths.
	$(PYTHON) scripts/validate_site_bundle.py --root "$(SITE_OUTPUT_DIR)"

site-serve: ## Serve the generated site bundle locally. Set SITE_OPEN=1 to open the browser.
	extra_args=""; \
	if [[ "$(SITE_OPEN)" == "1" || "$(SITE_OPEN)" == "true" ]]; then \
		extra_args="--open"; \
	fi; \
	$(PYTHON) scripts/serve_showcase.py \
		--dir "$(SITE_OUTPUT_DIR)" \
		--bind "$(SITE_BIND)" \
		--port "$(SITE_PORT)" \
		$$extra_args
