# Makefile for MovieFinderAI

.PHONY: install test test-unit test-api test-analysis lint format clean run run-api run-api-prod run-dashboard upload

# Virtualenv Python (falls back to python3)
PYTHON ?= .venv/bin/python
PIPX := $(shell [ -x .venv/bin/pip ] && echo .venv/bin/pip || echo pip3)

install:
	@echo "Installing dependencies..."
	uv pip install --python .venv/bin/python -r pyproject.toml


test: test-unit test-api test-analysis
	@echo "All tests passed!"


test-unit:
	@echo "Running unit tests..."
	$(PYTHON) -m pytest tests/unit/ -v


test-api:
	@echo "Running API tests..."
	$(PYTHON) -m pytest tests/api/ -v


test-analysis:
	@echo "Running analysis tests..."
	$(PYTHON) -m pytest tests/analysis/ -v


test-cov:
	@echo "Running tests with coverage..."
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=html


lint:
	@echo "Running linting..."
	$(PYTHON) -m mypy src/ --ignore-missing-imports


format:
	@echo "Running code formatter..."
	$(PYTHON) -m black src/ tests/


clean:
	@echo "Cleaning..."
	rm -rf .pytest_cache .coverage coverage.html htmlcov
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +


run:
	@echo "Running Gradio app..."
	$(PYTHON) -m src.interfaces.pages.gradio_app


run-dashboard:
	@echo "Running Streamlit dashboard..."
	$(PYTHON) -m streamlit run src/interfaces/pages/dashboard.py


run-api:
	@echo "Running FastAPI..."
	$(PYTHON) -m uvicorn src.interfaces.api.main:app --host 0.0.0.0 --port 8000 --reload


run-api-prod:
	@echo "Running FastAPI in production mode..."
	$(PYTHON) -m uvicorn src.interfaces.api.main:app --host 0.0.0.0 --port 8000 --workers 4


upload:
	@echo "Uploading CSVs to Chroma Cloud (CSVs/ -> movies_docs)..."
	$(PYTHON) -m scripts.upload_to_chroma --popular-first


help:
	@echo "Available commands:"
	@echo "  make install        - Install dependencies"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-api       - Run API tests only"
	@echo "  make test-analysis  - Run analysis tests only"
	@echo "  make test-cov       - Run tests with coverage"
	@echo "  make lint           - Run linting"
	@echo "  make format         - Format code"
	@echo "  make clean          - Clean cache files"
	@echo "  make run            - Run Gradio app"
	@echo "  make run-dashboard  - Run Streamlit dashboard"
	@echo "  make run-api        - Run FastAPI in development mode"
	@echo "  make run-api-prod   - Run FastAPI in production mode"
