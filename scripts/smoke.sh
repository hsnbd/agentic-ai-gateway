#!/usr/bin/env bash
set -euo pipefail
# Defaults to `make dev` (port 4000); the Docker image listens on 8000.
base_url="${AIGATEWAY_URL:-http://localhost:4000}"
api_key="${MASTER_KEY:-sk-gateway-master-change-me}"
curl -fsS "$base_url/healthz" >/dev/null
curl -fsS "$base_url/readyz" >/dev/null
curl -fsS "$base_url/v1/chat/completions" -H 'Content-Type: application/json' -H "Authorization: Bearer $api_key" -d "{\"model\":\"${SMOKE_MODEL:-llama3.1}\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with OK\"}],\"max_tokens\":8}" >/dev/null
printf 'Smoke checks passed.\n'
