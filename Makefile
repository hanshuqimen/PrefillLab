.PHONY: install test lint frontend dev build

install:
	pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check .
	ruff format --check .
	mypy prefilllab

frontend:
	cd frontend && npm install && npm run build

dev:
	uvicorn prefilllab.api.app:app --reload

build:
	python -m build
