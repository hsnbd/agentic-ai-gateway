"""Tests for the composition root.

The builder's job is to decide what the gateway does when a subsystem is
missing. Those decisions are policy, not plumbing, so they are pinned here:
an unavailable cache must degrade to a working gateway, while unavailable
auth must refuse to start.
"""

from __future__ import annotations

import builtins
from types import SimpleNamespace
from typing import Any

import pytest

from app.core.builder import build_pipeline
from app.providers.registry import ProviderRegistry


def _settings(**overrides: Any) -> SimpleNamespace:
    base = {
        "guardrails_enabled": True,
        "guardrails_config_path": "config/guardrails.yaml",
        "cache_enabled": True,
        "pricing_config_path": "config/pricing.yaml",
        "routing_strategy": "priority",
        "circuit_breaker_threshold": 5,
        "circuit_breaker_cooldown_seconds": 30.0,
        "max_retries": 2,
        "retry_base_delay_seconds": 0.5,
        "retry_max_delay_seconds": 8.0,
        "max_fallbacks": 3,
        "cache_similarity_threshold": 0.95,
        "cache_ttl_seconds": 3600,
        "cache_embedding_model": "nomic-embed-text",
        "cache_embedding_dimensions": 768,
        "cache_index_name": "aigw:cache:idx",
        "cache_max_temperature": 0.3,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


class _FakeState:
    def __init__(self, *, redis: Any = None, **setting_overrides: Any) -> None:
        self.settings = _settings(**setting_overrides)
        self.registry = ProviderRegistry(self.settings)  # type: ignore[arg-type]
        self.redis = redis
        self.db = None
        self.components: dict[str, Any] = {}
        self.breaker = None
        self.router = None


def test_pipeline_has_auth_first() -> None:
    """Unauthenticated work must cost nothing, so auth runs before anything
    that could touch a provider or the cache."""
    pipeline = build_pipeline(_FakeState())  # type: ignore[arg-type]
    assert pipeline.pre_stages[0].name == "auth"


def test_guardrails_precede_cache() -> None:
    """A blocked prompt should never even be looked up in the cache."""
    pipeline = build_pipeline(_FakeState(redis=object()))  # type: ignore[arg-type]
    names = [s.name for s in pipeline.pre_stages]
    assert names.index("guardrails_input") < names.index("cache")


def test_observability_runs_last() -> None:
    """The request log must capture the final, post-guardrail response."""
    pipeline = build_pipeline(_FakeState())  # type: ignore[arg-type]
    assert pipeline.post_stages[-1].name == "observability"


def test_cache_skipped_without_redis() -> None:
    """No Redis is a degraded mode, not an outage. Traffic still flows."""
    pipeline = build_pipeline(_FakeState(redis=None))  # type: ignore[arg-type]
    names = [s.name for s in pipeline.pre_stages]
    assert "cache" not in names
    assert "auth" in names
    assert "cache_write" not in [s.name for s in pipeline.post_stages]


def test_cache_skipped_when_disabled() -> None:
    pipeline = build_pipeline(_FakeState(redis=object(), cache_enabled=False))  # type: ignore[arg-type]
    assert "cache" not in [s.name for s in pipeline.pre_stages]


def test_guardrails_skipped_when_disabled() -> None:
    pipeline = build_pipeline(_FakeState(guardrails_enabled=False))  # type: ignore[arg-type]
    names = [s.name for s in pipeline.pre_stages]
    assert "guardrails_input" not in names
    assert "guardrails_output" not in [s.name for s in pipeline.post_stages]


def test_bad_guardrail_config_degrades_rather_than_crashes() -> None:
    """A malformed policy file should not take the gateway down; it logs and
    continues without guardrails."""
    state = _FakeState(guardrails_config_path="config/does-not-exist.yaml")
    pipeline = build_pipeline(state)  # type: ignore[arg-type]
    assert "guardrails_input" not in [s.name for s in pipeline.pre_stages]
    assert "auth" in [s.name for s in pipeline.pre_stages]


def test_missing_auth_stage_is_fatal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serving unauthenticated traffic silently is worse than not booting."""
    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "app.auth.stage":
            raise ImportError("simulated missing auth module")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(RuntimeError, match="auth stage is unavailable"):
        build_pipeline(_FakeState())  # type: ignore[arg-type]


def test_executor_is_published_on_state() -> None:
    """Health endpoints and the console read breaker/router state, so the
    builder must expose them rather than keeping them private."""
    state = _FakeState()
    build_pipeline(state)  # type: ignore[arg-type]
    assert state.breaker is not None
    assert state.router is not None


def test_accounting_degrades_but_keeps_metrics() -> None:
    """Losing the price table should cost us cost rows, not observability."""
    state = _FakeState(pricing_config_path="config/nope.yaml")
    pipeline = build_pipeline(state)  # type: ignore[arg-type]
    assert pipeline.post_stages[-1].name == "observability"
    assert "usage" not in state.components
