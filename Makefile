# Makefile for ML Microstructure Signals

.PHONY: help install install-dev test lint format type-check clean demo dashboard docs

help: ## Show this help message
	@echo "ML Microstructure Signals - Available Commands:"
	@echo "=============================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install production dependencies
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage
	pytest tests/ --cov=ml_microstructure --cov-report=html --cov-report=term-missing

lint: ## Run linting
	ruff check .
	black --check .

format: ## Format code
	ruff check . --fix
	black .

type-check: ## Run type checking
	mypy ml_microstructure/

clean: ## Clean up generated files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf mlruns/
	rm -rf demo_output/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

demo: ## Run synthetic demo
	python demo.py

dashboard: ## Launch Streamlit dashboard
	streamlit run ml_microstructure/dashboards/streamlit_app.py

train: ## Train model with default config
	python -m ml_microstructure.pipeline.train config=configs/model/lgbm.yaml

predict: ## Make predictions (requires run_id)
	python -m ml_microstructure.pipeline.predict run_id=$(RUN_ID)

evaluate: ## Evaluate model (requires run_id)
	python -m ml_microstructure.pipeline.evaluate run_id=$(RUN_ID)

backtest: ## Run backtest (requires run_id)
	python -m ml_microstructure.backtest.run run_id=$(RUN_ID)

docs: ## Build documentation
	cd docs && make html

notebooks: ## Launch Jupyter notebooks
	jupyter notebook notebooks/

ci: ## Run CI pipeline locally
	make lint
	make type-check
	make test-cov

all: ## Run full pipeline: install, test, demo
	make install-dev
	make ci
	make demo

# Data download commands (placeholder)
download-lobster: ## Download LOBSTER data (placeholder)
	@echo "LOBSTER data download not implemented"
	@echo "Please download data manually from: https://lobsterdata.com/"

download-crypto: ## Download crypto data (placeholder)
	@echo "Crypto data download not implemented"
	@echo "Please download data manually from Kaggle or exchange APIs"

# Development helpers
setup: ## Initial setup for development
	make install-dev
	make clean
	@echo "Development environment ready!"

check: ## Quick check: lint + type-check
	make lint
	make type-check

# Example usage:
# make train
# make predict RUN_ID=abc123
# make evaluate RUN_ID=abc123
# make backtest RUN_ID=abc123



