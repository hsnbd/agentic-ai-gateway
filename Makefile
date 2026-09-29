.PHONY: help install dev test test-unit test-integration test-up test-down lint fmt typecheck check ui-install ui-dev ui-build ui-lint up down logs migrate e2e-install e2e-up e2e e2e-api e2e-ui e2e-down

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies into .venv
	uv venv --python 3.12
	uv pip install -e ".[dev]"

dev: ## Run the gateway with autoreload
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 4000

TEST_COMPOSE = docker compose -f tests/integration/docker-compose.test.yaml
E2E_COMPOSE = docker compose -f e2e/docker-compose.e2e.yaml

test: test-up ## Run unit + integration tests (starts Postgres and Redis Stack)
	uv run pytest

test-unit: ## Run unit tests only (no services needed)
	uv run pytest tests/unit -q

test-integration: test-up ## Run integration tests against real Postgres and Redis Stack
	uv run pytest tests/integration -q

test-up: ## Start the throwaway test datastores
	$(TEST_COMPOSE) up -d --wait

test-down: ## Stop the test datastores
	$(TEST_COMPOSE) down -v

lint: ## Lint with ruff
	uv run ruff check app tests scripts/fake_mcp_server.py

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

ui-lint: ## Type-check and lint the console
	cd ui && npx tsc --noEmit && npm run lint

up: ## Start Redis, Postgres, and the observability stack
	docker compose -f deploy/docker/compose.yaml up -d

down: ## Stop the local stack
	docker compose -f deploy/docker/compose.yaml down

logs: ## Tail the local stack logs
	docker compose -f deploy/docker/compose.yaml logs -f

migrate: ## Apply database migrations
	uv run alembic upgrade head

e2e-install: ## Install the Cucumber/Playwright E2E suite and its browser
	cd e2e && npm ci && npx playwright install --with-deps chromium

e2e-up: ## Build and start the full dockerised stack for E2E
	$(E2E_COMPOSE) up -d --build --wait

e2e: e2e-up ## Run every Cucumber scenario (API + console UI)
	cd e2e && npm test

e2e-api: e2e-up ## Run the API Cucumber scenarios only
	cd e2e && npm run test:api

e2e-ui: e2e-up ## Run the console UI Cucumber scenarios only
	cd e2e && npm run test:ui

e2e-down: ## Stop the E2E stack
	$(E2E_COMPOSE) down -v
