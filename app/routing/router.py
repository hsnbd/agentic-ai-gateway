"""Router: turns a model name into an ordered list of deployments to try.

The router filters by capability and health, applies the caller's virtual-key
allowlist, then hands the survivors to a strategy for ranking. The full ranked
list becomes the fallback chain, so failover never needs to re-route.
"""

from __future__ import annotations

from app.core.errors import NoHealthyDeploymentError, NotFoundError
from app.core.pipeline import RequestContext, RoutingDecision
from app.providers.base import Deployment
from app.providers.registry import ProviderRegistry
from app.routing.breaker import CircuitBreaker
from app.routing.strategies import get_strategy


class Router:
    def __init__(
        self,
        registry: ProviderRegistry,
        breaker: CircuitBreaker,
        default_strategy: str = "priority",
    ) -> None:
        self._registry = registry
        self._breaker = breaker
        self._default_strategy = default_strategy

    def route(self, ctx: RequestContext) -> tuple[RoutingDecision, list[Deployment]]:
        """Return the primary decision plus the ordered fallback chain."""
        request = ctx.request
        rejected: dict[str, str] = {}

        candidates = self._collect_candidates(ctx, rejected)
        total_considered = len(candidates) + len(rejected)

        capable = [d for d in candidates if d.capabilities.supports(request)]
        for dep in candidates:
            if dep not in capable:
                rejected[dep.id] = "capabilities do not satisfy the request"

        if not capable:
            raise NoHealthyDeploymentError(
                f"No deployment can serve {request.model!r} with the requested features",
                details={"rejected": rejected},
            )

        healthy = [d for d in capable if self._breaker.is_available(d.id)]
        if not healthy:
            # Every deployment is circuit-broken. Rather than fail outright,
            # try them anyway: an outage that healed is better discovered by a
            # real request than by refusing to serve traffic.
            for dep in capable:
                rejected[dep.id] = "circuit breaker open (retrying anyway)"
            healthy = capable

        strategy_name = request.routing_strategy or self._default_strategy
        strategy = get_strategy(strategy_name)
        chain = strategy.order(healthy, request, self._breaker)
        primary, reason = strategy.select(healthy, request, self._breaker)

        # `select` and `order` are independent calls, so keep them consistent.
        if chain[0].id != primary.id:
            chain = [primary] + [d for d in chain if d.id != primary.id]

        decision = RoutingDecision(
            deployment=primary,
            strategy=strategy.name,
            reason=reason,
            candidates_considered=total_considered,
            candidates_rejected=rejected,
        )
        return decision, chain

    def _collect_candidates(
        self, ctx: RequestContext, rejected: dict[str, str]
    ) -> list[Deployment]:
        """Deployments for the requested model plus any explicit fallbacks."""
        request = ctx.request
        candidates = self._registry.deployments_for(request.model)

        for fallback_model in request.fallbacks:
            try:
                candidates.extend(self._registry.deployments_for(fallback_model))
            except NotFoundError:
                rejected[fallback_model] = "configured fallback model is not registered"

        # De-duplicate while preserving order.
        seen: set[str] = set()
        unique: list[Deployment] = []
        for dep in candidates:
            if dep.id not in seen:
                seen.add(dep.id)
                unique.append(dep)

        return self._apply_key_policy(ctx, unique, rejected)

    def _apply_key_policy(
        self,
        ctx: RequestContext,
        candidates: list[Deployment],
        rejected: dict[str, str],
    ) -> list[Deployment]:
        key = ctx.virtual_key
        if key is None:
            return candidates

        allowed = getattr(key, "allowed_models", None)
        blocked = getattr(key, "blocked_models", None)

        result: list[Deployment] = []
        for dep in candidates:
            if allowed and dep.model_name not in allowed and dep.id not in allowed:
                rejected[dep.id] = "model not in the virtual key allowlist"
                continue
            if blocked and (dep.model_name in blocked or dep.id in blocked):
                rejected[dep.id] = "model is blocked for this virtual key"
                continue
            result.append(dep)

        if not result and candidates:
            raise NoHealthyDeploymentError(
                f"Virtual key is not permitted to use {ctx.request.model!r}",
                details={"rejected": rejected},
            )
        return result
