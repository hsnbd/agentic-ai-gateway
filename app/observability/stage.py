from __future__ import annotations

from typing import Any, Literal

from app.accounting.pricing import PriceTable
from app.accounting.usage import UsageService
from app.core.pipeline import RequestContext
from app.core.schemas import ChatResponse, FinishReason
from app.observability import metrics
from app.observability.logging import bind_request_context, get_logger, reset_request_context
from app.observability.tracing import current_trace_id

logger = get_logger(__name__)


class ObservabilityStage:
    name = "observability"

    def __init__(self, usage_service: UsageService | None = None) -> None:
        self.usage_service = usage_service

    def _usage_service(self, ctx: RequestContext) -> UsageService:
        if self.usage_service is not None:
            return self.usage_service
        existing = ctx.state.components.get("usage_service")
        if isinstance(existing, UsageService):
            return existing
        service = UsageService(
            ctx.state.db,
            ctx.state.redis,
            PriceTable(ctx.state.settings.pricing_config_path),
            settings=ctx.state.settings,
        )
        ctx.state.components["usage_service"] = service
        return service

    async def finalize(self, ctx: RequestContext, response: ChatResponse) -> ChatResponse:
        service = self._usage_service(ctx)
        deployment = ctx.routing.deployment if ctx.routing is not None else None
        provider = deployment.provider if deployment is not None else response.provider or "unknown"
        trace_id = ctx.trace_id or current_trace_id()
        context_tokens = bind_request_context(ctx.request_id, trace_id)
        status = (
            "error"
            if any(choice.finish_reason == FinishReason.ERROR for choice in response.choices)
            else "success"
        )
        try:
            await service.record(ctx, response)
            duration = response.latency_ms or ctx.elapsed_ms()
            if getattr(ctx.state.settings, "metrics_enabled", True):
                metrics.record_request(
                    response.model, provider, status, ctx.dialect, duration / 1000
                )
                metrics.record_tokens(
                    response.model,
                    provider,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    response.usage.cached_tokens,
                )
                key_id = ctx.key_id or getattr(ctx.virtual_key, "id", None)
                metrics.record_cost(response.model, provider, key_id, response.cost_usd or 0.0)
                cache_enabled = getattr(ctx.state.settings, "cache_enabled", True)
                cache_result: Literal["hit", "miss", "skip"] = (
                    "skip"
                    if ctx.request.no_cache or not cache_enabled
                    else "hit"
                    if ctx.cache_hit
                    else "miss"
                )
                metrics.record_cache(cache_result, ctx.cost_saved_usd)
                if ctx.time_to_first_token_ms is not None:
                    metrics.record_ttft(response.model, provider, ctx.time_to_first_token_ms / 1000)
                if status == "error":
                    error = (
                        ctx.errors[-1]
                        if ctx.errors
                        else RuntimeError("response finish reason error")
                    )
                    metrics.record_error(response.model, provider, _error_code(error))
                for error in ctx.errors:
                    metrics.record_retry(provider, _error_code(error))
                if ctx.fallback_used:
                    reason = _error_code(ctx.errors[-1]) if ctx.errors else "fallback"
                    metrics.record_fallback("unknown", provider, reason)
                if ctx.guardrail_flagged:
                    metrics.record_guardrail(
                        ctx.request.guardrail_policy or "default", "request", "flagged"
                    )
            logger.info(
                "request completed",
                request_id=ctx.request_id,
                model=response.model,
                provider=provider,
                status=status,
                latency_ms=duration,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                cached_tokens=response.usage.cached_tokens,
                cost_usd=response.cost_usd,
                cache_hit=ctx.cache_hit,
                attempt_count=max(ctx.attempt_count, 1),
                fallback_used=ctx.fallback_used,
                stage_timings=ctx.stage_timings,
                trace_id=trace_id,
            )
        finally:
            reset_request_context(context_tokens)
        return response


def _error_code(error: Exception) -> str:
    code: Any = getattr(error, "code", None)
    return str(getattr(code, "value", code) or type(error).__name__)
