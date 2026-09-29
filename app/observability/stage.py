from __future__ import annotations

from typing import Any

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
            await _charge_spend(ctx, response.cost_usd or 0.0)
            if not ctx.cache_hit:
                await _count_tokens(ctx, response.usage.total_tokens)
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
                # Only count lookups the cache stage actually performed. Retries,
                # fallbacks, and time-to-first-token are recorded by the
                # executor where they happen, so they are not repeated here.
                if ctx.cache_result is not None:
                    metrics.record_cache(ctx.cache_result, ctx.cost_saved_usd)
                if status == "error":
                    error = (
                        ctx.errors[-1]
                        if ctx.errors
                        else RuntimeError("response finish reason error")
                    )
                    metrics.record_error(response.model, provider, _error_code(error))
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

    async def on_failure(self, ctx: RequestContext, error: Exception) -> None:
        """Log and meter a request that never produced a response."""
        service = self._usage_service(ctx)
        await service.record_error(ctx, error)
        deployment = ctx.routing.deployment if ctx.routing is not None else None
        provider = deployment.provider if deployment is not None else "unknown"
        if getattr(ctx.state.settings, "metrics_enabled", True):
            metrics.record_request(
                ctx.model, provider, "error", ctx.dialect, ctx.elapsed_ms() / 1000
            )
            metrics.record_error(ctx.model, provider, _error_code(error))
        logger.info(
            "request failed",
            request_id=ctx.request_id,
            model=ctx.model,
            provider=provider,
            error_code=_error_code(error),
            latency_ms=ctx.elapsed_ms(),
            attempt_count=max(ctx.attempt_count, 1),
        )


async def _count_tokens(ctx: RequestContext, tokens: int) -> None:
    """Feed actual usage into the key and deployment tokens-per-minute windows."""
    from app.auth.ratelimit import TokenWindow
    from app.auth.stage import deployment_tpm_counter, key_tpm_counter

    if ctx.state.redis is None or tokens <= 0:
        return
    window = TokenWindow(ctx.state.redis)
    key = ctx.virtual_key
    if key is not None and getattr(key, "tpm_limit", None) is not None:
        await window.add(key_tpm_counter(key.id), tokens)
    deployment = ctx.routing.deployment if ctx.routing is not None else None
    if deployment is not None and deployment.tpm_limit is not None:
        await window.add(deployment_tpm_counter(deployment.id), tokens)


async def _charge_spend(ctx: RequestContext, cost_usd: float) -> None:
    """Charge a virtual key (and its team) so budgets are enforceable."""
    key_id = ctx.key_id
    if key_id is None or key_id == "master" or cost_usd <= 0:
        return
    from app.auth.quotas import QuotaService

    try:
        await QuotaService(ctx.state.db, ctx.state.redis).record_spend(
            key_id, ctx.team_id, cost_usd
        )
    except Exception:
        logger.exception("could not record spend", request_id=ctx.request_id, key_id=key_id)


def _error_code(error: Exception) -> str:
    code: Any = getattr(error, "code", None)
    return str(getattr(code, "value", code) or type(error).__name__)
