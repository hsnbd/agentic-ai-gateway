from __future__ import annotations

import logging
import re
import sys
from collections.abc import Mapping, MutableMapping
from contextvars import Token
from typing import Any, cast

import structlog

REDACT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(?<![A-Za-z0-9])sk-ant-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?i)(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{8,}"),
    re.compile(r"(?<![A-Za-z0-9])ghp_[A-Za-z0-9]{8,}"),
    re.compile(r"(?<![A-Za-z0-9])AKIA[A-Z0-9]{16}"),
    re.compile(r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/-]+=*"),
    re.compile(r"(?i)(\bauthorization\s*[:=]\s*).+"),
)
_SENSITIVE_KEYS = {"api_key", "authorization", "password", "token", "secret", "key_hash"}


def redact(value: str) -> str:
    result = value
    for pattern in REDACT_PATTERNS:
        if pattern.groups:
            result = pattern.sub(r"\1[REDACTED]", result)
        else:
            result = pattern.sub("[REDACTED]", result)
    return result


def redact_mapping(d: dict[str, Any]) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: "[REDACTED]" if str(key).lower() in _SENSITIVE_KEYS else clean(item)
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, tuple):
            return tuple(clean(item) for item in value)
        if isinstance(value, str):
            return redact(value)
        return value

    return cast(dict[str, Any], clean(d))


def _redact_event(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    return redact_mapping(dict(event_dict))


def configure_logging(settings: Any) -> None:
    renderer: structlog.typing.Processor
    if settings.log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer()

    timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp")
    shared_processors: list[structlog.typing.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _redact_event,
    ]
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[structlog.stdlib.ProcessorFormatter.remove_processors_meta, renderer],
    )
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    structlog.configure(
        processors=[*shared_processors, structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


def bind_request_context(request_id: str, trace_id: str | None = None) -> Mapping[str, Token[Any]]:
    values: dict[str, Any] = {"request_id": request_id, "trace_id": trace_id}
    return structlog.contextvars.bind_contextvars(**values)


def reset_request_context(tokens: Mapping[str, Token[Any]]) -> None:
    structlog.contextvars.reset_contextvars(**tokens)
