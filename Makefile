.PHONY: help install dev test test-unit lint fmt typecheck check ui-install ui-dev ui-build up down logs migrate

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies into .venv
	uv venv --python 3.12
	uv pip install -e ".[dev]"

dev: ## Run the gateway with autoreload
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 4000

test: ## Run the full test suite
	uv run pytest

test-unit: ## Run unit tests only
	uv run pytest tests/unit -q

lint: ## Lint with ruff
	uv run ruff check app tests

fmt: ## Format and autofix with ruff
	uv run ruff format app tests
	uv run ruff check --fix app tests

typecheck: ## Type-check with mypy
	uv run mypy app

check: lint typecheck test ## Lint, type-check, and test

ui-install: ## Install console dependencies
	cd ui && npm install

ui-dev: ## Run the console dev server
	cd ui && npm run dev

ui-build: ## Build the console into app/ui_static
	cd ui && npm run build

up: ## Start Redis, Postgres, and the observability stack
	docker compose -f deploy/docker/compose.yaml up -d

down: ## Stop the local stack
	docker compose -f deploy/docker/compose.yaml down

logs: ## Tail the local stack logs
	docker compose -f deploy/docker/compose.yaml logs -f

migrate: ## Apply database migrations
	uv run alembic upgrade head
