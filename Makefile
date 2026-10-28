.PHONY: help install dev dev-deps test test-unit test-integration coverage ui-test test-up test-down lint fmt fmt-check typecheck check bench bench-check bench-baseline eval hub hub-mcp hub-seed hub-run hub-test hub-eval ui-install ui-dev ui-build ui-lint up down logs migrate e2e-install e2e-up e2e e2e-api e2e-ui e2e-down

help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Install Python dependencies into .venv
	uv venv --python 3.12
	uv pip install -e ".[dev]"

dev: dev-deps ## Start the dependencies, migrate, then run the gateway with autoreload
	uv run aigateway migrate
	uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 4000

dev-deps: ## Start Postgres, Redis, and Ollama (with the embedding models) for `make dev`
	./scripts/dev-deps.sh

TEST_COMPOSE = docker compose -f tests/integration/docker-compose.test.yaml
E2E_COMPOSE = docker compose -f e2e/docker-compose.e2e.yaml

test: test-up ## Run unit + integration tests (starts Postgres and Redis Stack)
	uv run pytest

test-unit: ## Run unit tests only (no services needed)
	uv run pytest tests/unit -q

test-integration: test-up ## Run integration tests against real Postgres and Redis Stack
	uv run pytest tests/integration -q

coverage: test-up ## Unit + integration with the 100% line and branch coverage gate
	uv run pytest --cov --cov-report=term-missing:skip-covered --cov-report=html

test-up: ## Start the throwaway test datastores
	$(TEST_COMPOSE) up -d --wait

test-down: ## Stop the test datastores
	$(TEST_COMPOSE) down -v

lint: ## Lint with ruff
	uv run ruff check app tests scripts evals

fmt: ## Format and autofix with ruff
	uv run ruff format app tests scripts evals
	uv run ruff check --fix app tests scripts evals

fmt-check: ## Fail if any file is not ruff-formatted
	uv run ruff format --check app tests scripts evals

typecheck: ## Type-check with mypy
	uv run mypy app evals

check: lint fmt-check typecheck coverage ui-test ## Lint, format, type-check, and run every gated suite

ui-install: ## Install console dependencies
	cd ui && npm install

ui-dev: ## Run the console dev server
	cd ui && npm run dev

ui-build: ## Build the console into app/ui_static
	cd ui && npm run build

ui-test: ## Unit-test the console with its coverage gate
	cd ui && npm run test:coverage

ui-lint: ## Type-check and lint the console
	cd ui && npx tsc --noEmit && npm run lint

up: ## Start Redis, Postgres, and the observability stack
	docker compose -f deploy/docker/compose.yaml up -d

down: ## Stop the local stack
	docker compose -f deploy/docker/compose.yaml down

logs: ## Tail the local stack logs
	docker compose -f deploy/docker/compose.yaml logs -f

migrate: ## Apply database migrations (Alembic)
	uv run aigateway migrate

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

BENCH_BASELINE ?= bench/baseline

bench: e2e-up ## Run every benchmark against the e2e stack; results in bench/results/<timestamp>
	uv run python scripts/bench.py

bench-check: e2e-up ## Benchmark and fail on regressions against bench/baseline
	uv run python scripts/bench.py --baseline $(BENCH_BASELINE)

EVAL_DATASET ?= bench/datasets/eval_gateway.yaml
EVAL_OUT ?= bench/results/eval

eval: ## Score answer quality with DeepEval (needs AIGW_BASE_URL, AIGW_API_KEY, EVAL_MODEL) and gate it
	uv run python -m evals $(EVAL_DATASET) --out $(EVAL_OUT)
	uv run python scripts/bench_compare.py $(EVAL_OUT) $(EVAL_OUT)

bench-baseline: e2e-up ## Re-record bench/baseline from a fresh run (review the diff before committing)
	rm -rf $(BENCH_BASELINE)
	uv run python scripts/bench.py --out bench/.baseline-run
	mv "$$(ls -d bench/.baseline-run/*/ | tail -1)" $(BENCH_BASELINE)
	rm -rf bench/.baseline-run

# ---- IIT information hub (apps/iit-hub): a LangGraph app built on the gateway
HUB = cd apps/iit-hub &&
HUB_PORT ?= 8100
HUB_MCP_PORT ?= 4300
GATEWAY_URL ?= http://localhost:4000

hub: ## Run the whole IIT hub: mock IIT system, gateway provisioning, and the hub (needs the gateway running)
	@curl -sf $(GATEWAY_URL)/healthz >/dev/null || { echo "Gateway not reachable at $(GATEWAY_URL); start it first (make dev)"; exit 1; }
	@cd apps/iit-hub && uv sync -q && \
	  { uv run python -m mock_iit.mcp_server --port $(HUB_MCP_PORT) > mcp.log 2>&1 & MCP=$$!; } && \
	  trap 'kill $$MCP 2>/dev/null' EXIT INT TERM && \
	  for i in $$(seq 30); do nc -z 127.0.0.1 $(HUB_MCP_PORT) 2>/dev/null && break; sleep 0.5; done && \
	  uv run --env-file ../../.env python -m mock_iit.seed \
	    --gateway $(GATEWAY_URL) --mcp-url http://127.0.0.1:$(HUB_MCP_PORT)/mcp $(if $(HUB_RESET),--reset,) && \
	  echo "IIT hub on http://localhost:$(HUB_PORT) (Ctrl+C stops it and the mock IIT system)" && \
	  uv run uvicorn iit_hub.api:app --port $(HUB_PORT)

hub-mcp: ## Run the hub's mock IIT student information system (MCP server on :4300)
	$(HUB) uv run python -m mock_iit.mcp_server --port 4300

hub-seed: ## Provision the running gateway for the hub (team, MCP server, collections, role keys)
	$(HUB) uv run --env-file ../../.env python -m mock_iit.seed --reset

hub-run: ## Run the IIT hub on http://localhost:8100
	$(HUB) uv run uvicorn iit_hub.api:app --port $(HUB_PORT)

hub-test: ## Test the IIT hub
	$(HUB) uv run pytest

hub-eval: ## DeepEval answer quality of the running hub (needs AIGW_API_KEY and EVAL_JUDGE_MODEL)
	HUB_URL=http://localhost:$(HUB_PORT) uv run python -m evals apps/iit-hub/evals/dataset.yaml --out bench/results/hub-eval
