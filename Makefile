.PHONY: help install run test lint clean docker-up docker-down mock-data

help:
	@echo "Speaklar RAG System - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install dependencies"
	@echo "  make mock-data        Generate mock Bangla product dataset"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-up        Start Redis container"
	@echo "  make docker-down      Stop Redis container"
	@echo ""
	@echo "Development:"
	@echo "  make run              Run API server"
	@echo "  make test             Run pytest suite"
	@echo "  make lint             Run ruff + black formatting"
	@echo "  make format           Auto-format code with black"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            Remove cache and build artifacts"

install:
	pip install -r requirements.txt

mock-data:
	python data/generate_mock_data.py

docker-up:
	docker-compose up -d redis
	@echo "Redis started on 0.0.0.0:6379"

docker-down:
	docker-compose down

run:
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

test:
	pytest tests/ -v --cov=. --cov-report=term-missing

lint:
	ruff check . && black --check .

format:
	black . && ruff check --fix .

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info 2>/dev/null || true

