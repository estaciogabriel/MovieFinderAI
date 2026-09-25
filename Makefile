# Makefile for MovieFinderAI

.PHONY: install test test-unit test-api lint format clean run run-api

# Default Python
PYTHON ?= python3
PIP ?= pip3

install:
	@echo "Installing dependencies..."
	$(PIP) install -r requirements.txt


test: test-unit test-api
	@echo "All tests passed!"


test-unit:
	@echo "Running unit tests..."
	$(PYTHON) -m pytest tests/unit/ -v


test-api:
	@echo "Running API tests..."
	$(PYTHON) -m pytest tests/api/ -v


test-cov:
	@echo "Running tests with coverage..."
	$(PYTHON) -m pytest tests/ --cov=api --cov=movies_knowledge_base --cov-report=html


lint:
	@echo "Running linting..."
	$(PYTHON) -m mypy api/ movies_knowledge_base/ --ignore-missing-imports


format:
	@echo "Running code formatter..."
	$(PYTHON) -m black api/ movies_knowledge_base/ tests/


clean:
	@echo "Cleaning..."
	rm -rf __pycache__ */__pycache__ */*/__pycache__ .pytest_cache .coverage coverage.html


run:
	@echo "Running application..."
	$(PYTHON) app.py


run-api:
	@echo "Running FastAPI..."
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload


run-api-prod:
	@echo "Running FastAPI in production mode..."
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4


help:
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Run all tests"
	@echo "  make test-unit  - Run unit tests only"
	@echo "  make test-api   - Run API tests only"
	@echo "  make test-cov   - Run tests with coverage"
	@echo "  make lint      - Run linting"
	@echo "  make format    - Format code"
	@echo "  make clean     - Clean cache files"
	@echo "  make run       - Run original Gradio app"
	@echo "  make run-api    - Run FastAPI in development mode"
	@echo "  make run-api-prod - Run FastAPI in production mode"
