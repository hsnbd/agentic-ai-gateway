FROM node:22-alpine AS ui-build
WORKDIR /ui
COPY ui/package*.json ./
RUN npm ci
COPY ui/ ./
RUN npm run build

FROM python:3.12-slim AS runtime
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PATH="/opt/venv/bin:$PATH"
WORKDIR /app
RUN apt-get update && apt-get install --no-install-recommends -y ca-certificates && rm -rf /var/lib/apt/lists/* && pip install --no-cache-dir uv
COPY pyproject.toml uv.lock ./
RUN uv venv /opt/venv && uv export --frozen --no-dev --extra rag-docs --no-hashes --no-emit-project > /tmp/requirements.txt && uv pip install --python /opt/venv/bin/python -r /tmp/requirements.txt && rm -rf /root/.cache/uv /tmp/requirements.txt
# Bake tiktoken's BPE files into the image: otherwise the first token count
# for an OpenAI model downloads them at request time (and cannot, offline).
ENV TIKTOKEN_CACHE_DIR=/app/.cache/tiktoken
RUN python -c "import tiktoken; [tiktoken.get_encoding(name) for name in ('o200k_base', 'cl100k_base')]"
COPY app/ ./app/
COPY config/ ./config/
# Vite's outDir is ../app/ui_static (so a local `npm run build` lands where the
# gateway serves from), which inside the ui-build stage resolves to
# /app/ui_static — not /ui/dist.
COPY --from=ui-build /app/ui_static ./app/ui_static
RUN groupadd --system aigateway && useradd --system --gid aigateway --home-dir /app --no-create-home aigateway && chown -R aigateway:aigateway /app /opt/venv
USER aigateway
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=3)"
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
