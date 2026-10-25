#!/usr/bin/env bash
# Start what `make dev` needs: Postgres and Redis in Docker, plus Ollama with the
# embedding models. A native Ollama (`ollama serve`, the macOS app) is used when
# one is installed, since it gets the GPU; otherwise the compose container runs.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -f .env ]]; then cp .env.example .env; fi
compose=(docker compose --env-file .env -f deploy/docker/compose.yaml)

env_value() {
  local value
  value="$(grep -E "^$1=" .env | tail -1 | cut -d= -f2- || true)"
  printf '%s' "${value:-$2}"
}

if ! docker info >/dev/null 2>&1; then
  if [[ "$(uname)" == "Darwin" ]]; then
    printf 'Starting Docker Desktop...\n'
    open -a Docker
    for _ in {1..60}; do
      if docker info >/dev/null 2>&1; then break; fi
      sleep 2
    done
  fi
  docker info >/dev/null 2>&1 || { printf 'Docker is not running; start it and retry.\n' >&2; exit 1; }
fi

"${compose[@]}" up -d --wait postgres redis

ollama_url="$(env_value OLLAMA_BASE_URL http://localhost:11434)"
ollama_up() { curl -fsS "$ollama_url/api/tags" >/dev/null 2>&1; }

if ! ollama_up; then
  if [[ "$ollama_url" != *localhost* && "$ollama_url" != *127.0.0.1* ]]; then
    printf 'Ollama at %s is unreachable; local models will fail.\n' "$ollama_url" >&2
  else
    if command -v ollama >/dev/null 2>&1; then
      printf 'Starting native Ollama (log: .ollama-dev.log)...\n'
      nohup ollama serve >.ollama-dev.log 2>&1 &
    else
      "${compose[@]}" up -d ollama
    fi
    for _ in {1..30}; do
      if ollama_up; then break; fi
      sleep 1
    done
    ollama_up || { printf 'Ollama did not start at %s.\n' "$ollama_url" >&2; exit 1; }
  fi
fi

if ollama_up; then
  installed="$(curl -fsS "$ollama_url/api/tags")"
  for model in "$(env_value CACHE_EMBEDDING_MODEL nomic-embed-text)" \
               "$(env_value RAG_EMBEDDING_MODEL nomic-embed-text)"; do
    if [[ "$installed" != *"\"$model"* ]]; then
      printf 'Pulling %s...\n' "$model"
      curl -fsS "$ollama_url/api/pull" -d "{\"model\":\"$model\",\"stream\":false}" >/dev/null
      installed="$installed \"$model"
    fi
  done
fi
printf 'Dev dependencies are up.\n'
