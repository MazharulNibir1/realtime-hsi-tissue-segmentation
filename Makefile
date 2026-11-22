# Hyperspectral tissue classification — common tasks.
# Usage: `make setup` once, then `make lint test`.

PYTHON ?= python3.11
VENV   ?= .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python

.DEFAULT_GOAL := help
.PHONY: help setup lint format test precommit clean data data-dry data-verify

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

$(PY):
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install --upgrade pip

setup: $(PY)  ## Create venv, install package (editable) + dev tools, install git hooks
	$(PY) -m pip install -e ".[dev]"
	@if [ -d .git ]; then $(BIN)/pre-commit install; else echo "No .git yet: run '$(BIN)/pre-commit install' after git init"; fi

lint:  ## Ruff + Black (check only)
	$(BIN)/ruff check .
	$(BIN)/black --check .

format:  ## Auto-fix lint + format
	$(BIN)/ruff check --fix .
	$(BIN)/black .

test:  ## Run unit tests
	$(BIN)/pytest

precommit:  ## Run all pre-commit hooks on all files
	$(BIN)/pre-commit run --all-files

clean:  ## Remove caches (keeps venv, data, runs)
	find . -type d \( -name __pycache__ -o -name .pytest_cache -o -name .ruff_cache \) -prune -exec rm -rf {} +
	rm -rf build dist *.egg-info src/*.egg-info

data-dry:  ## Build the subset manifest and print download size (no cube download)
	$(PY) scripts/download_data.py --dry-run

data:  ## Download the approved HeiPorSPECTRAL subset (~23 GB, resumable)
	$(PY) scripts/download_data.py

data-verify:  ## CRC32-verify every downloaded file
	$(PY) scripts/download_data.py --verify
