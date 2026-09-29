from __future__ import annotations

import importlib

import pytest

from app.config.settings import Settings
from app.observability import metrics, tracing
from app.observability.logging import redact, redact_mapping


def test_redact_secrets_and_authorization_values() -> None:
    text = (
        "key sk-1234567890 and sk-ant-1234567890 ghp_1234567890 "
        "AKIA1234567890ABCDEF Bearer abc.def.ghi Authorization: Basic secret-value"
    )
    result = redact(text)
    for secret in (
        "sk-1234567890",
        "sk-ant-1234567890",
        "ghp_1234567890",
        "AKIA1234567890ABCDEF",
        "abc.def.ghi",
        "Basic secret-value",
    ):
        assert secret not in result


def test_redact_nested_sensitive_keys_regardless_of_value_type() -> None:
    result = redact_mapping(
        {
            "nested": {
                "api_key": {"secret": "nested"},
                "Authorization": "Bearer xyz",
                "public": ["token sk-1234567890", {"key_hash": "abcdef"}],
                "password": None,
            }
        }
    )
    nested = result["nested"]
    assert nested["api_key"] == "[REDACTED]"
    assert nested["Authorization"] == "[REDACTED]"
    assert nested["password"] == "[REDACTED]"
    assert nested["public"][0] == "token [REDACTED]"
    assert nested["public"][1]["key_hash"] == "[REDACTED]"


def test_metrics_helpers_and_module_reimport_are_safe() -> None:
    metrics.record_request("test-model", "test-provider", "success", "openai", 0.2)
    metrics.record_error("test-model", "test-provider", "test-error")
    metrics.record_tokens("test-model", "test-provider", 4, 2, 1)
    metrics.record_cost("test-model", "test-provider", "internal-key-id", 0.01)
    metrics.record_cache("hit", 0.02)
    metrics.record_cache("miss")
    metrics.record_cache("skip")
    metrics.record_ttft("test-model", "test-provider", 0.1)
    metrics.record_fallback("provider-a", "provider-b", "unavailable")
    metrics.record_retry("test-provider", "timeout")
    metrics.set_provider_health("test-provider", "deployment-1", True)
    metrics.record_guardrail("policy", "request", "allow")
    metrics.set_active_requests("test-model", 1)
    metrics.record_rate_limit("key")

    reloaded = importlib.reload(metrics)
    reloaded.record_request("test-model", "test-provider", "success", "openai", 0.0)
    assert reloaded.REQUESTS is not None


def test_virtual_key_metric_label_is_internal_and_bounded() -> None:
    metrics.record_cost("model", "provider", "x" * 1000, 0.01)
    assert len(metrics._key_label("x" * 1000)) == 64
    assert metrics._key_label(None) == "anonymous"


@pytest.mark.asyncio
async def test_tracing_helpers_are_noops_when_disabled() -> None:
    tracing._enabled = False
    tracing._tracer = None
    assert tracing.current_trace_id() is None
    async with tracing.span("disabled-span", attribute="value") as current:
        assert current is None
    tracing.configure_tracing(Settings(tracing_enabled=False), object())
    assert tracing.current_trace_id() is None


def test_metric_helper_contract_signatures() -> None:
    import inspect
    from contextlib import AbstractContextManager

    expected = {
        "record_retry": ("provider", "error_code"),
        "record_fallback": ("from_provider", "to_provider", "reason"),
        "set_provider_health": ("provider", "deployment", "healthy"),
        "record_ttft": ("model", "provider", "seconds"),
        "record_request": ("model", "provider", "status", "dialect", "duration_seconds"),
        "record_tokens": ("model", "provider", "prompt", "completion", "cached"),
        "record_cost": ("model", "provider", "virtual_key", "usd"),
        "record_cache": ("result", "cost_saved_usd"),
        "record_guardrail": ("policy", "phase", "action"),
        "record_rate_limit": ("scope",),
        "track_active": ("model",),
    }
    for name, parameter_names in expected.items():
        helper = getattr(metrics, name)
        assert callable(helper)
        signature = inspect.signature(helper)
        assert tuple(signature.parameters) == parameter_names
        assert all(
            parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in signature.parameters.values()
        )
        if name == "track_active":
            assert signature.return_annotation == "AbstractContextManager[None]"
        else:
            assert signature.return_annotation == "None"
    assert isinstance(metrics.track_active("contract-test"), AbstractContextManager)
    with metrics.track_active("contract-test"):
        assert metrics.ACTIVE_REQUESTS.labels("contract-test")._value.get() >= 1


def test_logger_works_without_configure_logging() -> None:
    from app.observability.logging import get_logger

    log = get_logger(__name__)
    log.warning("provider_retry", request_id="req-test", attempt=1)
    log.info("request_complete", request_id="req-test")
    log.error("request_error", request_id="req-test")
