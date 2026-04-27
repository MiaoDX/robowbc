SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

PYTHON ?= python3
CARGO ?= cargo
MDBOOK_VERSION ?= 0.4.40
MDBOOK_BIN_DIR ?= $(CURDIR)/.cache/bin
MDBOOK ?= $(MDBOOK_BIN_DIR)/mdbook
MUJOCO_VERSION ?= 3.6.0

SITE_OUTPUT_DIR ?= /tmp/robowbc-site
SITE_BIND ?= 127.0.0.1
SITE_PORT ?= 8000
SITE_OPEN ?= 0
SITE_BROWSER_POLICY ?= gear_sonic
MUJOCO_DOWNLOAD_DIR ?= $(CURDIR)/.cache/mujoco
MUJOCO_DYNAMIC_LINK_DIR ?= $(abspath $(MUJOCO_DOWNLOAD_DIR))/mujoco-$(MUJOCO_VERSION)/lib
SHOWCASE_MUJOCO_GL ?= egl
SHOWCASE_PYOPENGL_PLATFORM ?= $(SHOWCASE_MUJOCO_GL)
ROBOWBC_BINARY ?= $(CURDIR)/target/debug/robowbc
SITE_PYTHON_DEPS ?= numpy joblib onnxruntime==1.24.4 pyyaml mujoco Pillow
PYTHON_SDK_DIST_DIR ?= $(CURDIR)/dist
PYTHON_SDK_MUJOCO_DOWNLOAD_DIR ?= $(CURDIR)/.cache/mujoco
PYTHON_SDK_TARGET_DIR ?= $(CURDIR)/target/python-sdk-wheel

.DEFAULT_GOAL := help

.PHONY: help toolchain build build-release check test sim-feature-test clippy fmt fmt-check rust-doc mdbook-install docs-book docs verify smoke models-public site-python-deps site-render-check benchmark-robowbc benchmark-official benchmark-summary benchmark-nvidia site showcase-verify site-smoke site-browser-smoke site-serve-check site-serve python-sdk-deps python-sdk-build python-sdk-install python-sdk-smoke python-sdk-verify ci

help: ## Show available targets and useful variables.
	@awk 'BEGIN {FS = ":.*## "; print "Targets:"} /^[a-zA-Z0-9_.-]+:.*## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@printf "\nVariables:\n"
	@printf "  %-20s %s\n" "PYTHON" "$(PYTHON)"
	@printf "  %-20s %s\n" "MDBOOK" "$(MDBOOK)"
	@printf "  %-20s %s\n" "SITE_OUTPUT_DIR" "$(SITE_OUTPUT_DIR)"
	@printf "  %-20s %s\n" "MUJOCO_VERSION" "$(MUJOCO_VERSION)"
	@printf "  %-20s %s\n" "MUJOCO_DOWNLOAD_DIR" "$(MUJOCO_DOWNLOAD_DIR)"
	@printf "  %-20s %s\n" "MUJOCO_DYNAMIC_LINK_DIR" "$(MUJOCO_DYNAMIC_LINK_DIR)"
	@printf "  %-20s %s\n" "SHOWCASE_MUJOCO_GL" "$(SHOWCASE_MUJOCO_GL)"
	@printf "  %-20s %s\n" "SHOWCASE_PYOPENGL_PLATFORM" "$(SHOWCASE_PYOPENGL_PLATFORM)"
	@printf "  %-20s %s\n" "PYTHON_SDK_DIST_DIR" "$(PYTHON_SDK_DIST_DIR)"
	@printf "  %-20s %s\n" "PYTHON_SDK_MUJOCO_DOWNLOAD_DIR" "$(PYTHON_SDK_MUJOCO_DOWNLOAD_DIR)"
	@printf "  %-20s %s\n" "PYTHON_SDK_TARGET_DIR" "$(PYTHON_SDK_TARGET_DIR)"
	@printf "  %-20s %s\n" "SITE_PORT" "$(SITE_PORT)"
	@printf "  %-20s %s\n" "SITE_OPEN" "$(SITE_OPEN)"
	@printf "  %-20s %s\n" "SITE_BROWSER_POLICY" "$(SITE_BROWSER_POLICY)"

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

sim-feature-test: ## Run the feature-enabled MuJoCo sim transport tests with the runtime/plugin env wired.
	MUJOCO_DOWNLOAD_DIR="$(abspath $(MUJOCO_DOWNLOAD_DIR))" \
	MUJOCO_DYNAMIC_LINK_DIR="$(MUJOCO_DYNAMIC_LINK_DIR)" \
	LD_LIBRARY_PATH="$(MUJOCO_DYNAMIC_LINK_DIR):$${LD_LIBRARY_PATH:-}" \
	$(CARGO) test -p robowbc-sim --features mujoco-auto-download

clippy: ## Run clippy with warnings promoted to errors.
	$(CARGO) clippy --workspace --all-targets -- -D warnings

fmt: ## Format the Rust workspace.
	$(CARGO) fmt --all

fmt-check: ## Verify Rust formatting without changing files.
	$(CARGO) fmt --all -- --check

rust-doc: ## Build Rust API docs.
	$(CARGO) doc --workspace --no-deps

mdbook-install: ## Download the pinned mdBook binary used by CI into ./.cache/bin.
	mkdir -p "$(MDBOOK_BIN_DIR)"
	if [[ ! -x "$(MDBOOK)" ]]; then \
		curl -sSL "https://github.com/rust-lang/mdBook/releases/download/v$(MDBOOK_VERSION)/mdbook-v$(MDBOOK_VERSION)-x86_64-unknown-linux-gnu.tar.gz" \
			| tar -xz --directory "$(MDBOOK_BIN_DIR)"; \
	fi

docs-book: mdbook-install ## Build the mdBook docs.
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

site-render-check: ## Verify the MuJoCo offscreen renderer works before building screenshot-bearing proof packs.
	MUJOCO_GL="$(SHOWCASE_MUJOCO_GL)" \
	PYOPENGL_PLATFORM="$(SHOWCASE_PYOPENGL_PLATFORM)" \
	$(PYTHON) scripts/check_mujoco_headless.py

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
	MUJOCO_GL="$(SHOWCASE_MUJOCO_GL)" \
	PYOPENGL_PLATFORM="$(SHOWCASE_PYOPENGL_PLATFORM)" \
	MUJOCO_DOWNLOAD_DIR="$(MUJOCO_DOWNLOAD_DIR)" \
	$(PYTHON) scripts/build_site.py \
		--repo-root . \
		--robowbc-binary "$(ROBOWBC_BINARY)" \
		--output-dir "$(SITE_OUTPUT_DIR)"

showcase-verify: ## Run the same showcase build + bundle validation path used in CI.
	$(MAKE) site-python-deps
	$(MAKE) site-render-check
	$(MAKE) models-public
	$(MAKE) site
	$(MAKE) site-smoke

site-smoke: ## Validate the generated site bundle layout and embedded playback paths.
	$(PYTHON) scripts/validate_site_bundle.py --root "$(SITE_OUTPUT_DIR)"

site-browser-smoke: site-smoke ## Run the optional headless browser lag-selector smoke test for one policy page.
	$(PYTHON) scripts/site_browser_smoke.py \
		--root "$(SITE_OUTPUT_DIR)" \
		--policy "$(SITE_BROWSER_POLICY)" \
		--bind "$(SITE_BIND)"

site-serve-check: ## Start the local site server briefly to confirm it boots, then stop it automatically.
	status=0; \
	timeout --signal=INT 2s $(MAKE) --no-print-directory site-serve \
		SITE_OUTPUT_DIR="$(SITE_OUTPUT_DIR)" \
		SITE_BIND="$(SITE_BIND)" \
		SITE_PORT="$(SITE_PORT)" \
		SITE_OPEN=0 || status=$$?; \
	if [[ $$status -ne 0 && $$status -ne 124 && $$status -ne 130 ]]; then \
		exit $$status; \
	fi

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

python-sdk-deps: ## Install Python build dependencies for local SDK verification.
	$(PYTHON) -m pip install "maturin>=1.4,<2.0"

python-sdk-build: python-sdk-deps ## Build the local RoboWBC Python wheel with an absolute MuJoCo cache path.
	mkdir -p "$(PYTHON_SDK_DIST_DIR)" "$(abspath $(PYTHON_SDK_MUJOCO_DOWNLOAD_DIR))"
	rm -f "$(PYTHON_SDK_DIST_DIR)"/robowbc-*.whl
	rm -rf "$(PYTHON_SDK_TARGET_DIR)"
	MUJOCO_DOWNLOAD_DIR="$(abspath $(PYTHON_SDK_MUJOCO_DOWNLOAD_DIR))" \
	CARGO_TARGET_DIR="$(PYTHON_SDK_TARGET_DIR)" \
	$(PYTHON) -m maturin build --release --out "$(PYTHON_SDK_DIST_DIR)" -i "$(PYTHON)"

python-sdk-install: python-sdk-build ## Install the freshly built local RoboWBC Python wheel.
	wheel="$$(ls -1t "$(PYTHON_SDK_DIST_DIR)"/robowbc-*.whl | head -n 1)"; \
	$(PYTHON) -m pip install --force-reinstall "$$wheel"

python-sdk-smoke: ## Run the installed RoboWBC Python SDK smoke test.
	$(PYTHON) scripts/python_sdk_smoke.py

python-sdk-verify: ## Build, install, and smoke-test the RoboWBC Python SDK locally.
	$(MAKE) python-sdk-install
	$(MAKE) python-sdk-smoke

ci: ## Run the same local validation entry points GitHub CI relies on.
	$(MAKE) verify
	$(MAKE) sim-feature-test
	$(MAKE) docs
	$(MAKE) python-sdk-verify
	$(MAKE) showcase-verify
