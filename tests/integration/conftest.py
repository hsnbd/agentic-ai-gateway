"""Shared fixtures for end-to-end integration tests.

These tests exercise the *real* app against *real* datastores: the real
FastAPI routers, the real pipeline built by the real composition root, real
Postgres, and real Redis Stack (vector search for the semantic cache and RAG).
Only the upstream LLM providers are faked, because we will not call paid APIs
in CI.

Start the datastores first:

    docker compose -f tests/integration/docker-compose.test.yaml up -d --wait

Point elsewhere with ``AIGW_TEST_DATABASE_URL`` / ``AIGW_TEST_REDIS_URL``.
Every test starts from an empty schema and an empty Redis.
"""

from __future__ import annotations

import asyncio
import hashlib
import math
import os
import re
import tempfile
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
import redis
from fastapi.testclient import TestClient

from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    Choice,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingVector,
    Message,
    StreamChunk,
    Usage,
)
from app.providers.base import Capabilities, Deployment, Pricing, Provider

MASTER_KEY = "sk-integration-master-key"
ADMIN_EMAIL = "admin@integration.test"
ADMIN_PASSWORD = "integration-admin-password"
JWT_SECRET = "integration-jwt-secret-that-is-long-enough-for-hs256"
EMBED_DIM = 32

TEST_DATABASE_URL = os.environ.get(
    "AIGW_TEST_DATABASE_URL",
    "postgresql+asyncpg://aigateway:aigateway@localhost:55432/aigateway_test",
)
# RediSearch indexes only work on logical database 0, so the suite owns a
# whole Redis instance rather than sharing a numbered database.
TEST_REDIS_URL = os.environ.get("AIGW_TEST_REDIS_URL", "redis://localhost:56379/0")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        if "tests/integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)


def hashed_embedding(text: str, dims: int = EMBED_DIM) -> list[float]:
    """Deterministic bag-of-words vector: similar texts land close together."""
    vector = [0.0] * dims
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        bucket = int(hashlib.sha256(word.encode()).hexdigest(), 16) % dims
        vector[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


class FakeProvider(Provider):
    """A provider whose behaviour each test can program.

    Defaults to succeeding. Set `fail_times` to make the first N calls raise,
    which is how the retry and fallback tests drive the resilient executor.
    """

    name = "fake"

    def __init__(
        self,
        *,
        reply: str = "fake response",
        fail_times: int = 0,
        error: Exception | None = None,
        latency: float = 0.0,
    ) -> None:
        self.reply = reply
        self.fail_times = fail_times
        self.error = error
        self.latency = latency
        self.calls = 0
        self.stream_calls = 0
        self.embed_calls = 0
        self.seen_requests: list[ChatRequest] = []

    def capabilities(self, deployment: Deployment) -> Capabilities:
        return Capabilities(
            streaming=True,
            tools=True,
            vision=False,
            json_mode=True,
            embeddings=True,
            max_context_tokens=128_000,
        )

    def pricing(self, deployment: Deployment) -> Pricing:
        return Pricing(input_per_mtok=1.0, output_per_mtok=2.0)

    def _maybe_fail(self) -> None:
        if self.calls <= self.fail_times:
            raise self.error or RuntimeError("fake provider failure")

    async def chat(self, request: ChatRequest, deployment: Deployment) -> ChatResponse:
        self.calls += 1
        self.seen_requests.append(request)
        if self.latency:
            await asyncio.sleep(self.latency)
        self._maybe_fail()
        return ChatResponse(
            model=request.model,
            provider=self.name,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=self.reply),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    def stream(self, request: ChatRequest, deployment: Deployment) -> AsyncIterator[StreamChunk]:
        async def _gen() -> AsyncIterator[StreamChunk]:
            self.stream_calls += 1
            self.calls += 1
            self.seen_requests.append(request)
            if self.latency:
                await asyncio.sleep(self.latency)
            self._maybe_fail()
            for token in self.reply.split():
                yield StreamChunk(model=request.model, provider=self.name, content=token + " ")
            yield StreamChunk(
                model=request.model,
                provider=self.name,
                content="",
                finish_reason="stop",
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        return _gen()

    async def embed(self, request: EmbeddingRequest, deployment: Deployment) -> EmbeddingResponse:
        self.embed_calls += 1
        return EmbeddingResponse(
            model=deployment.model_name,
            provider=self.name,
            data=[
                EmbeddingVector(index=i, embedding=hashed_embedding(text))
                for i, text in enumerate(request.input)
            ],
            usage=Usage(
                prompt_tokens=len(request.input),
                completion_tokens=0,
                total_tokens=len(request.input),
            ),
        )


MODELS_YAML = """
model_list:
  - model_name: test-model
    params:
      provider: fake
      model: fake-1
      api_key: unused
    priority: 1
    capabilities: {tools: true, json_mode: true, max_context_tokens: 128000}
    pricing: {input_per_mtok: 1000.0, output_per_mtok: 2000.0}
  - model_name: test-model
    params:
      provider: fake-backup
      model: fake-2
      api_key: unused
    priority: 2
    capabilities: {tools: true, json_mode: true, max_context_tokens: 128000}
    pricing: {input_per_mtok: 1000.0, output_per_mtok: 2000.0}
  - model_name: other-model
    params:
      provider: fake
      model: fake-other
      api_key: unused
  - model_name: embed-model
    params:
      provider: fake
      model: fake-embed
      api_key: unused
    capabilities: {chat: false, embeddings: true}
aliases:
  gpt-4o: test-model
"""


def _reset_datastores() -> None:
    """Drop every table and every Redis key/index so each test starts empty."""
    from sqlalchemy.ext.asyncio import create_async_engine

    from app.db.models import Base

    async def _reset_db() -> None:
        engine = create_async_engine(TEST_DATABASE_URL)
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
        finally:
            await engine.dispose()

    asyncio.run(_reset_db())

    client = redis.Redis.from_url(TEST_REDIS_URL)
    try:
        for index in client.execute_command("FT._LIST"):
            client.execute_command("FT.DROPINDEX", index)
        client.flushall()
    finally:
        client.close()


@pytest.fixture(scope="session", autouse=True)
def _datastores_available() -> None:
    """Fail fast, with instructions, when the test datastores are not running."""
    try:
        client = redis.Redis.from_url(TEST_REDIS_URL, socket_connect_timeout=2)
        client.ping()
        client.execute_command("FT._LIST")
        client.close()
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.exit(
            f"Redis Stack not reachable at {TEST_REDIS_URL} ({exc}). Run: docker compose "
            "-f tests/integration/docker-compose.test.yaml up -d --wait",
            returncode=2,
        )
    try:
        _reset_datastores()
    except Exception as exc:  # pragma: no cover - environment guard
        pytest.exit(
            f"Postgres not reachable at {TEST_DATABASE_URL} ({exc}). Run: docker compose "
            "-f tests/integration/docker-compose.test.yaml up -d --wait",
            returncode=2,
        )


@pytest.fixture
def models_yaml() -> str:
    """Override in a module to boot the gateway with a different catalogue."""
    return MODELS_YAML


@pytest.fixture
def models_config(models_yaml: str) -> Iterator[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(models_yaml)
        path = handle.name
    yield path
    os.unlink(path)


@pytest.fixture
def extra_env() -> dict[str, str]:
    """Override in a module (or parametrize) to change gateway settings."""
    return {}


@pytest.fixture
def primary() -> FakeProvider:
    return FakeProvider(reply="primary answer")


@pytest.fixture
def backup() -> FakeProvider:
    provider = FakeProvider(reply="backup answer")
    provider.name = "fake-backup"
    return provider


@pytest.fixture
def gateway_env(
    monkeypatch: pytest.MonkeyPatch, models_config: str, extra_env: dict[str, str]
) -> dict[str, str]:
    env = {
        "DATABASE_URL": TEST_DATABASE_URL,
        "REDIS_URL": TEST_REDIS_URL,
        "MODELS_CONFIG_PATH": models_config,
        "MASTER_KEY": MASTER_KEY,
        "JWT_SECRET": JWT_SECRET,
        "BOOTSTRAP_ADMIN_EMAIL": ADMIN_EMAIL,
        "BOOTSTRAP_ADMIN_PASSWORD": ADMIN_PASSWORD,
        "CACHE_ENABLED": "false",
        "CACHE_EMBEDDING_MODEL": "embed-model",
        "CACHE_EMBEDDING_DIMENSIONS": str(EMBED_DIM),
        "RAG_EMBEDDING_MODEL": "embed-model",
        "RAG_EMBEDDING_DIMENSIONS": str(EMBED_DIM),
        "RETRY_BASE_DELAY_SECONDS": "0",
        "RETRY_MAX_DELAY_SECONDS": "0",
        "LOG_FORMAT": "console",
        **extra_env,
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return env


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    gateway_env: dict[str, str],
    primary: FakeProvider,
    backup: FakeProvider,
) -> Iterator[TestClient]:
    """Boot the real app, on real datastores, with fake providers in the registry."""
    _reset_datastores()

    from app.config.settings import get_settings
    from app.providers import registry as registry_module

    get_settings.cache_clear()

    def fake_register(self: Any, http_client: Any) -> None:
        self._providers[primary.name] = primary
        self._providers[backup.name] = backup

    monkeypatch.setattr(registry_module.ProviderRegistry, "_register_providers", fake_register)

    from app.main import create_app

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    get_settings.cache_clear()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {MASTER_KEY}"}


@pytest.fixture
def admin_token(client: TestClient) -> str:
    response = client.post(
        "/admin/api/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD}
    )
    assert response.status_code == 200, response.text
    token: str = response.json()["access_token"]
    return token


@pytest.fixture
def admin_headers(admin_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {admin_token}"}


def chat_body(content: str = "Hello", model: str = "test-model", **extra: Any) -> dict[str, Any]:
    return {"model": model, "messages": [{"role": "user", "content": content}], **extra}


def create_virtual_key(
    client: TestClient, admin_headers: dict[str, str], **fields: Any
) -> tuple[str, dict[str, Any]]:
    """Create a virtual key through the admin API; returns (secret, key record)."""
    response = client.post(
        "/admin/api/keys", json={"name": "integration", **fields}, headers=admin_headers
    )
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body["key"], body


def request_logs(client: TestClient, admin_headers: dict[str, str], **params: Any) -> list[Any]:
    response = client.get("/admin/api/logs", params=params, headers=admin_headers)
    assert response.status_code == 200, response.text
    items: list[Any] = response.json()["items"]
    return items


def metric_value(client: TestClient, name: str, **labels: str) -> float:
    """Sum a Prometheus sample from /metrics across series matching `labels`."""
    from prometheus_client.parser import text_string_to_metric_families

    total = 0.0
    for family in text_string_to_metric_families(client.get("/metrics").text):
        for sample in family.samples:
            if sample.name == name and all(sample.labels.get(k) == v for k, v in labels.items()):
                total += sample.value
    return total


VIEWER_EMAIL = "viewer@integration.test"
VIEWER_PASSWORD = "integration-viewer-password"


@pytest.fixture
def viewer_headers(client: TestClient, admin_headers: dict[str, str]) -> dict[str, str]:
    """A read-only console user."""
    created = client.post(
        "/admin/api/users",
        json={"email": VIEWER_EMAIL, "password": VIEWER_PASSWORD, "role": "viewer"},
        headers=admin_headers,
    )
    assert created.status_code == 201, created.text
    login = client.post(
        "/admin/api/auth/login", json={"email": VIEWER_EMAIL, "password": VIEWER_PASSWORD}
    )
    assert login.status_code == 200, login.text
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


@pytest.fixture(scope="session")
def fake_mcp_url() -> Iterator[str]:
    """A real MCP server over streamable HTTP, on a free local port."""
    import socket
    import threading
    import time

    import uvicorn

    from scripts.fake_mcp_server import create_app as create_mcp_app

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_mcp_app(), host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:  # pragma: no cover - environment guard
            raise RuntimeError("fake MCP server did not start")
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=5)
