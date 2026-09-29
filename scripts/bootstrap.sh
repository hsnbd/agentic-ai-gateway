#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -f .env ]]; then cp .env.example .env; fi
docker compose --env-file .env -f deploy/docker/compose.yaml up -d
for _ in {1..60}; do
  if curl -fsS http://localhost:8000/healthz >/dev/null; then break; fi
  sleep 2
done
curl -fsS http://localhost:8000/healthz >/dev/null
cache_model="$(grep -E '^CACHE_EMBEDDING_MODEL=' .env | cut -d= -f2- || true)"
cache_model="${cache_model:-nomic-embed-text}"
rag_model="$(grep -E '^RAG_EMBEDDING_MODEL=' .env | cut -d= -f2- || true)"
rag_model="${rag_model:-nomic-embed-text}"
chat_model="$(awk '/provider: ollama/{found=1} found && /model:/{print $2; exit}' config/models.yaml)"
for model in "$cache_model" "$rag_model" "$chat_model"; do
  [[ -n "$model" ]] && docker compose --env-file .env -f deploy/docker/compose.yaml exec -T ollama ollama pull "$model"
done
printf 'Gateway is healthy.\n'
