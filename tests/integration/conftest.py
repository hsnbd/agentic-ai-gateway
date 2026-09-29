"""Shared fixtures for end-to-end integration tests.

These tests exercise the *real* app: the real FastAPI routers, the real
pipeline built by the real composition root, and the real auth/guardrail/
routing code. Only two things are faked:

  * the upstream providers, because we will not call paid APIs in CI; and
  * Redis, because the semantic cache needs a vector-search server.

Everything between the HTTP request and the provider call is the genuine
article, which is the point: these tests are what catch wiring mistakes that
unit tests with mocked neighbours cannot see.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.core.schemas import (
    ChatRequest,
    ChatResponse,
    Choice,
    EmbeddingResponse,
    Message,
    StreamChunk,
    Usage,
)
from app.providers.base import Capabilities, Deployment, Pricing, Provider

MASTER_KEY = "sk-integration-master-key"


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

    async def chat(
        self, request: ChatRequest, deployment: Deployment
    ) -> ChatResponse:
        self.calls += 1
        self.seen_requests.append(request)
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

    def stream(
        self, request: ChatRequest, deployment: Deployment
    ) -> AsyncIterator[StreamChunk]:
        async def _gen() -> AsyncIterator[StreamChunk]:
            self.stream_calls += 1
            self.calls += 1
            self.seen_requests.append(request)
            self._maybe_fail()
            for token in self.reply.split():
                yield StreamChunk(
                    model=request.model,
                    provider=self.name,
                    content=token + " ",
                )
            yield StreamChunk(
                model=request.model,
                provider=self.name,
                content="",
                finish_reason="stop",
                usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
            )

        return _gen()

    async def embed(
        self, texts: list[str], deployment: Deployment
    ) -> EmbeddingResponse:
        self.calls += 1
        return EmbeddingResponse(
            model=deployment.model_name,
            provider=self.name,
            embeddings=[[0.1] * 8 for _ in texts],
            usage=Usage(prompt_tokens=len(texts), completion_tokens=0, total_tokens=len(texts)),
        )


MODELS_YAML = """
model_list:
  - model_name: test-model
    params:
      provider: fake
      model: fake-1
      api_key: unused
    priority: 1
  - model_name: test-model
    params:
      provider: fake-backup
      model: fake-2
      api_key: unused
    priority: 2
aliases:
  gpt-4o: test-model
"""


@pytest.fixture
def models_config() -> Iterator[str]:
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as handle:
        handle.write(MODELS_YAML)
        path = handle.name
    yield path
    os.unlink(path)


@pytest.fixture
def primary() -> FakeProvider:
    return FakeProvider(reply="primary answer")


@pytest.fixture
def backup() -> FakeProvider:
    provider = FakeProvider(reply="backup answer")
    provider.name = "fake-backup"
    return provider


@pytest.fixture
def client(
    monkeypatch: pytest.MonkeyPatch,
    models_config: str,
    primary: FakeProvider,
    backup: FakeProvider,
    tmp_path: Any,
) -> Iterator[TestClient]:
    """Boot the real app with fake providers swapped into the registry."""
    db_path = tmp_path / "integration.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{db_path}")
    monkeypatch.setenv("MODELS_CONFIG_PATH", models_config)
    monkeypatch.setenv("MASTER_KEY", MASTER_KEY)
    # No Redis in CI: the gateway must degrade to a working, uncached gateway.
    monkeypatch.setenv("CACHE_ENABLED", "false")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/15")

    from app.config.settings import get_settings
    from app.providers import registry as registry_module

    get_settings.cache_clear()

    def fake_register(self: Any, http_client: Any) -> None:
        self._providers[primary.name] = primary
        self._providers[backup.name] = backup

    monkeypatch.setattr(
        registry_module.ProviderRegistry, "_register_providers", fake_register
    )

    # Production migrates with Alembic. Tests create the schema inline, on the
    # app's own event loop, by extending startup rather than reaching into it.
    from app.db.session import Database
    from app.main import create_app

    original_startup = Database.startup

    async def startup_with_schema(self: Database) -> None:
        await original_startup(self)
        await self.create_all()

    monkeypatch.setattr(Database, "startup", startup_with_schema)

    app = create_app()
    with TestClient(app) as test_client:
        yield test_client

    get_settings.cache_clear()


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {MASTER_KEY}"}
