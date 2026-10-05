.PHONY: install install-root lock venv reset-env clean-venv verify-arch test cov check-wheel lint fmt check pre-commit-install pre-commit-run help

# ── Tool selection ─────────────────────────────────────────────────────────────
# Override on the command line:  make install TOOL=uv  or  export TOOL=uv
TOOL ?= poetry

ifeq ($(TOOL),uv)
  _install      := uv sync --no-install-project
  _install_root := uv sync
  _lock         := uv lock
else
  POETRY ?= poetry
  _install      := $(POETRY) install --no-root
  _install_root := $(POETRY) install
  _lock         := $(POETRY) lock
endif

# Call the venv's binaries directly so targets work without activating it.
VENV := $(CURDIR)/.venv
PY := $(VENV)/bin/python

# ── Interpreter architecture ──────────────────────────────────────────────────
# uv keeps both macOS builds of CPython in its cache and will happily build the
# venv from whichever it already has, so an Apple Silicon Mac can end up on an
# x86_64 interpreter running under Rosetta: slower, and pinned to the older
# onnxruntime that still ships x86_64 wheels. The venv targets below name the
# build explicitly instead of letting uv choose.
#
# ARCH defaults to the host, so Intel Macs get a native x86_64 venv with no
# extra flags. Override it to cross-build deliberately — on Apple Silicon,
# `make reset-env ARCH=x86_64` gives a Rosetta venv for reproducing Intel-only
# bugs (needs Rosetta 2 installed).
UV ?= uv
PY_VERSION ?= 3.13
ARCH ?= $(shell uname -m)
# uv spells Apple Silicon "aarch64"; uname and platform.machine() say "arm64".
_UV_ARCH := $(patsubst arm64,aarch64,$(ARCH))

ifeq ($(shell uname -s),Darwin)
  PY_REQUEST ?= cpython-$(PY_VERSION)-macos-$(_UV_ARCH)-none
else
  PY_REQUEST ?= $(PY_VERSION)
endif

# Suites that run without loading the embedding model. Named once because both
# test and cov need the list; add new directories here.
FAST_TESTS := tests/engine/ tests/models/ tests/ports/ \
	tests/adapters/vector_store/ tests/adapters/file_source/ \
	tests/adapters/storage/ tests/adapters/retrieval/ \
	tests/adapters/embedder/test_make_embedder.py \
	tests/adapters/embedder/test_mps.py \
	tests/adapters/embedder/test_query_prefixes.py \
	tests/queries/ tests/indexing/ tests/retrieval/ \
	tests/test_cli.py \
	tests/test_exploration.py tests/test_exploration_cli.py

# ── Help ──────────────────────────────────────────────────────────────────────

help:
	@echo "Available targets:"
	@echo "  install             Install project dependencies (TOOL=poetry|uv)"
	@echo "  install-root        Install dependencies and the project package (TOOL=poetry|uv)"
	@echo "  lock                Generate or update the lock file (TOOL=poetry|uv)"
	@echo "  reset-env           Rebuild .venv from scratch on the native CPU (ARCH=x86_64|arm64)"
	@echo "  venv                Create or replace .venv only, without installing"
	@echo "  clean-venv          Delete .venv"
	@echo "  verify-arch         Report the venv's architecture and ONNX Runtime providers"
	@echo "  test                Run tests"
	@echo "  cov                 Run tests with coverage report"
	@echo "  check-wheel         Verify the built wheel carries the tags.scm files"
	@echo "  lint                Run Ruff lint checks"
	@echo "  fmt                 Run Ruff format checks"
	@echo "  check               Run lint and tests with coverage"
	@echo "  pre-commit-install  Install pre-commit hooks"
	@echo "  pre-commit-run      Run pre-commit hooks against all files"

# ── Dependencies ──────────────────────────────────────────────────────────────

install:
	$(_install)

install-root:
	$(_install_root)

lock:
	$(_lock)

# ── Environment ───────────────────────────────────────────────────────────────
# These targets always use uv, whatever TOOL is set to: Poetry consumes an
# interpreter but cannot fetch one, and choosing the interpreter is the whole
# point here.

clean-venv:
	rm -rf $(VENV)

# --clear replaces an existing .venv; uv refuses to overwrite one without it.
venv:
	$(UV) python install $(PY_REQUEST)
	$(UV) venv --clear --python $(PY_REQUEST)

# The one-shot rebuild: native interpreter, all dependency groups, then proof
# that the result matches the CPU. Re-activate the shell afterwards
# (`source .venv/bin/activate`) — the old venv this replaces is gone.
reset-env: venv
	$(UV) sync --all-groups
	@$(MAKE) --no-print-directory verify-arch

# A venv on the wrong architecture still runs, just slowly, so nothing fails
# loudly on its own. CoreMLExecutionProvider in the provider list is the sign
# the embedding model can reach the Apple Silicon GPU.
verify-arch:
	@$(PY) -c "import platform, sys; m = platform.machine(); \
	    print('interpreter :', sys.executable); \
	    print('architecture:', m, '(expected $(ARCH))'); \
	    sys.exit(0 if m == '$(ARCH)' else 1)" \
	    || { echo "ERROR: .venv does not match $(ARCH) — run 'make reset-env'."; exit 1; }
	@$(PY) -c "import onnxruntime as o; print('onnxruntime :', o.__version__); \
	    print('providers   :', ', '.join(o.get_available_providers()))" 2>/dev/null \
	    || echo "onnxruntime : not installed — run 'make reset-env'"

# ── Test ──────────────────────────────────────────────────────────────────────

test:
	$(PY) -m pytest $(FAST_TESTS) -v --no-cov

cov:
	$(PY) -m pytest $(FAST_TESTS) \
	    --cov=smritikosh \
	    --cov-report=term-missing \
	    --cov-report=html:htmlcov \
	    --cov-fail-under=90

# The language directories under smritikosh/queries/ are not packages, so the
# .scm files reach the wheel only through a package-data glob. Nothing in the
# test suite would notice if that glob stopped matching, hence this check.
check-wheel:
	@rm -rf dist/wheel-check
	@$(PY) -m build --wheel --outdir dist/wheel-check >/dev/null
	@expected=$$(find smritikosh/queries -name 'tags.scm' | wc -l | tr -d '[:space:]'); \
	found=$$(unzip -l dist/wheel-check/*.whl | grep -c 'queries/.*/tags\.scm'); \
	if [ "$$found" -ne "$$expected" ]; then \
	    echo "ERROR: wheel carries $$found tags.scm files; $$expected are in smritikosh/queries/."; \
	    echo "Check [tool.setuptools.package-data] in pyproject.toml."; \
	    rm -rf dist/wheel-check; exit 1; \
	fi; \
	rm -rf dist/wheel-check; \
	echo "Wheel carries all $$expected tags.scm files."

# ── Lint / Format ─────────────────────────────────────────────────────────────

lint:
	$(VENV)/bin/ruff check smritikosh/ tests/

fmt:
	$(VENV)/bin/ruff format smritikosh/ tests/

check: lint cov

# ── Pre-commit ────────────────────────────────────────────────────────────────

pre-commit-install:
	$(VENV)/bin/pre-commit install

pre-commit-run:
	$(VENV)/bin/pre-commit run --all-files
