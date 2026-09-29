from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from app.observability.logging import get_logger

logger = get_logger(__name__)
_enabled = False
_tracer: Any = None


def configure_tracing(settings: Any, app: Any) -> None:
    global _enabled, _tracer
    _enabled = False
    _tracer = None
    if not settings.tracing_enabled:
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter

        provider = TracerProvider(resource=Resource.create({"service.name": settings.app_name}))
        exporter = (
            OTLPSpanExporter(endpoint=settings.otlp_endpoint)
            if settings.otlp_endpoint
            else ConsoleSpanExporter()
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        FastAPIInstrumentor.instrument_app(app, tracer_provider=provider)
        _tracer = provider.get_tracer("aigateway")
        _enabled = True
    except Exception as exc:
        logger.warning("OpenTelemetry setup failed; tracing disabled", error=str(exc))


@asynccontextmanager
async def span(name: str, **attributes: Any) -> AsyncIterator[Any | None]:
    if not _enabled or _tracer is None:
        yield None
        return
    try:
        manager = _tracer.start_as_current_span(name, attributes=attributes)
        current = manager.__enter__()
    except Exception as exc:
        logger.warning("OpenTelemetry span failed", span_name=name, error=str(exc))
        yield None
        return
    try:
        yield current
    except BaseException:
        try:
            manager.__exit__(*sys.exc_info())
        except Exception as exc:
            logger.warning("OpenTelemetry span cleanup failed", span_name=name, error=str(exc))
        raise
    else:
        try:
            manager.__exit__(None, None, None)
        except Exception as exc:
            logger.warning("OpenTelemetry span cleanup failed", span_name=name, error=str(exc))


def current_trace_id() -> str | None:
    if not _enabled:
        return None
    try:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        if context.is_valid:
            return f"{context.trace_id:032x}"
    except Exception as exc:
        logger.warning("Could not read OpenTelemetry trace ID", error=str(exc))
    return None
