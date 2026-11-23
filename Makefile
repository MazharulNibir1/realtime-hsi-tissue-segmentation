# Hyperspectral tissue classification — common tasks.
# Usage: `make setup` once, then `make lint test`.

PYTHON ?= python3.11
VENV   ?= .venv
BIN    := $(VENV)/bin
PY     := $(BIN)/python

.DEFAULT_GOAL := help
.PHONY: help setup lint format test precommit clean data data-dry data-verify preprocess train train-smoke train-all evaluate export samples demo-models app

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

preprocess:  ## Bin + label the downloaded subset into data/processed (~10 GB)
	$(PY) scripts/preprocess_data.py

MODEL ?= unet
BANDS ?= full
SEED  ?= 0
train:  ## Train one model: make train MODEL=unet BANDS=snapshot16 SEED=0
	$(PY) scripts/train.py --model configs/models/$(MODEL).yaml --bands $(BANDS) --seed $(SEED)

train-smoke:  ## 1-epoch run of each model on a few images (pipeline check, writes runs/smoke)
	for m in pixel_mlp patch_cnn unet; do \
	  $(PY) scripts/train.py --model configs/models/$$m.yaml --bands snapshot16 --out runs/smoke --force \
	    --set train.epochs=1 train.limit_images=12 || exit 1; done

train-all:  ## T5: all models x {full, snapshot16} x seeds {0,1,2} (resumable, ~3.6 h on M3 Pro)
	$(PY) scripts/train_all.py

evaluate:  ## T6: test-set metrics, figures and results/results.md for all finished runs
	$(PY) scripts/evaluate.py

export:  ## T7: ONNX export (fp32/fp16/int8), parity, latency benchmark -> results/benchmark.md
	$(PY) scripts/export_benchmark.py

samples:  ## T8: demo samples (test pigs, CC BY 4.0) -> app/samples and web/public/samples
	$(PY) scripts/make_samples.py

demo-models:  ## Copy the chosen ONNX demo models into app/models and web/public/models
	$(PY) scripts/prepare_demo_models.py

app:  ## Run the Gradio demo locally (separate venv from app/requirements.txt, as on HF Spaces)
	test -x .venv-app/bin/python || ($(PYTHON) -m venv .venv-app && .venv-app/bin/pip install -r app/requirements.txt)
	cd app && ../.venv-app/bin/python app.py
