"""Routing strategies.

A strategy picks one deployment from the healthy candidates that can serve a
request. Strategies never perform I/O: they score what the registry and the
circuit breaker already know.
"""

from __future__ import annotations

import abc
import random
from typing import TYPE_CHECKING

from app.core.schemas import ChatRequest
from app.providers.base import Deployment

if TYPE_CHECKING:
    from app.routing.breaker import CircuitBreaker


class RoutingStrategy(abc.ABC):
    name: str = "base"

    @abc.abstractmethod
    def select(
        self,
        candidates: list[Deployment],
        request: ChatRequest,
        breaker: CircuitBreaker,
    ) -> tuple[Deployment, str]:
        """Return the chosen deployment and a human-readable reason."""

    def order(
        self,
        candidates: list[Deployment],
        request: ChatRequest,
        breaker: CircuitBreaker,
    ) -> list[Deployment]:
        """Full preference order, used to build the fallback chain.

        Default: the selected deployment first, then the rest in input order.
        """
        chosen, _ = self.select(candidates, request, breaker)
        return [chosen] + [d for d in candidates if d.id != chosen.id]


def _estimated_prompt_tokens(request: ChatRequest) -> int:
    """Cheap character-based estimate; good enough for cost ranking."""
    chars = sum(len(m.text()) for m in request.messages)
    return max(chars // 4, 1)


class LeastCostStrategy(RoutingStrategy):
    """Pick the cheapest deployment for the estimated shape of this request."""

    name = "least-cost"

    def select(
        self, candidates: list[Deployment], request: ChatRequest, breaker: CircuitBreaker
    ) -> tuple[Deployment, str]:
        prompt_tokens = _estimated_prompt_tokens(request)
        completion_tokens = request.max_tokens or 512

        scored = sorted(
            candidates,
            key=lambda d: (
                d.pricing.estimate(prompt_tokens, completion_tokens),
                -d.priority,
            ),
        )
        best = scored[0]
        cost = best.pricing.estimate(prompt_tokens, completion_tokens)
        return best, f"cheapest of {len(candidates)} candidates (~${cost:.6f} est.)"

    def order(
        self, candidates: list[Deployment], request: ChatRequest, breaker: CircuitBreaker
    ) -> list[Deployment]:
        prompt_tokens = _estimated_prompt_tokens(request)
        completion_tokens = request.max_tokens or 512
        return sorted(
            candidates,
            key=lambda d: (d.pricing.estimate(prompt_tokens, completion_tokens), -d.priority),
        )


class LowestLatencyStrategy(RoutingStrategy):
    """Pick the deployment with the best observed EWMA latency.

    Unmeasured deployments are tried first so that every deployment earns a
    sample instead of the first fast one winning forever.
    """

    name = "lowest-latency"

    def select(
        self, candidates: list[Deployment], request: ChatRequest, breaker: CircuitBreaker
    ) -> tuple[Deployment, str]:
        ordered = self.order(candidates, request, breaker)
        best = ordered[0]
        latency = breaker.health(best.id).ewma_latency_ms
        if latency <= 0:
            return best, "no latency samples yet; probing this deployment"
        return best, f"lowest observed latency ({latency:.0f} ms EWMA)"

    def order(
        self, candidates: list[Deployment], request: ChatRequest, breaker: CircuitBreaker
    ) -> list[Deployment]:
        def key(d: Deployment) -> tuple[int, float]:
            latency = breaker.health(d.id).ewma_latency_ms
            # (0, 0.0) sorts unmeasured deployments ahead of measured ones.
            return (0, 0.0) if latency <= 0 else (1, latency)

        return sorted(candidates, key=key)


class WeightedStrategy(RoutingStrategy):
    """Randomly distribute load in proportion to configured weights."""

    name = "weighted"

    def select(
        self, candidates: list[Deployment], request: ChatRequest, breaker: CircuitBreaker
    ) -> tuple[Deployment, str]:
        weights = [max(d.weight, 0) for d in candidates]
        total = sum(weights)
        if total <= 0:
            chosen = random.choice(candidates)
            return chosen, "all weights are zero; chose uniformly at random"

        chosen = random.choices(candidates, weights=weights, k=1)[0]
        share = max(chosen.weight, 0) / total
        return chosen, f"weighted random ({share:.0%} share)"

    def order(
        self, candidates: list[Deployment], request: ChatRequest, breaker: CircuitBreaker
    ) -> list[Deployment]:
        # Weighted sampling without replacement, so the fallback chain also
        # respects the configured distribution.
        pool = list(candidates)
        ordered: list[Deployment] = []
        while pool:
            weights = [max(d.weight, 0) or 1 for d in pool]
            pick = random.choices(pool, weights=weights, k=1)[0]
            ordered.append(pick)
            pool.remove(pick)
        return ordered


class PriorityStrategy(RoutingStrategy):
    """Strict preference order: the *lowest* `priority` number wins.

    `priority: 1` means "try this first", matching the convention used by DNS
    SRV records, nginx upstreams, and every other fallback list users have
    seen. Ties are broken by heavier weight, then by id for determinism.
    """

    name = "priority"

    def select(
        self, candidates: list[Deployment], request: ChatRequest, breaker: CircuitBreaker
    ) -> tuple[Deployment, str]:
        ordered = self.order(candidates, request, breaker)
        best = ordered[0]
        return best, f"priority {best.priority} (lowest wins)"

    def order(
        self, candidates: list[Deployment], request: ChatRequest, breaker: CircuitBreaker
    ) -> list[Deployment]:
        return sorted(candidates, key=lambda d: (d.priority, -d.weight, d.id))


class ConditionalStrategy(RoutingStrategy):
    """Route on request shape rather than on static configuration.

    In order: request `tags` matching deployment tags win (cheapest match);
    long prompts go to large-context deployments; tool use goes to
    tool-capable ones; short prompts prefer deployments tagged `cheap`.
    """

    name = "conditional"

    def __init__(self, long_context_threshold: int = 16000) -> None:
        self._long_context_threshold = long_context_threshold

    def select(
        self, candidates: list[Deployment], request: ChatRequest, breaker: CircuitBreaker
    ) -> tuple[Deployment, str]:
        prompt_tokens = _estimated_prompt_tokens(request)

        tagged = _tag_matches(candidates, request)
        if tagged:
            best, _ = LeastCostStrategy().select(tagged, request, breaker)
            shared = sorted(set(best.tags) & set(request.tags))
            return best, f"request tags {', '.join(shared)} matched deployment tags"

        if prompt_tokens >= self._long_context_threshold:
            large = [
                d for d in candidates if d.capabilities.max_context_tokens >= prompt_tokens * 2
            ]
            if large:
                best = max(large, key=lambda d: d.capabilities.max_context_tokens)
                return best, (
                    f"~{prompt_tokens} prompt tokens needs a large context window "
                    f"({best.capabilities.max_context_tokens})"
                )

        if request.requires_tools():
            tooled = [d for d in candidates if d.capabilities.tools]
            if tooled:
                best = min(
                    tooled,
                    key=lambda d: d.pricing.estimate(prompt_tokens, request.max_tokens or 512),
                )
                return best, "cheapest tool-capable deployment"

        cheap = [d for d in candidates if "cheap" in d.tags]
        if cheap and prompt_tokens < 2000:
            best = cheap[0]
            return best, f"short prompt (~{prompt_tokens} tokens) routed to a cheap model"

        return LeastCostStrategy().select(candidates, request, breaker)

    def order(
        self, candidates: list[Deployment], request: ChatRequest, breaker: CircuitBreaker
    ) -> list[Deployment]:
        chosen, _ = self.select(candidates, request, breaker)
        rest = [d for d in candidates if d.id != chosen.id]
        # Other tag matches are the first fallbacks, then everything else.
        tagged = {d.id for d in _tag_matches(rest, request)}
        return [
            chosen,
            *LeastCostStrategy().order([d for d in rest if d.id in tagged], request, breaker),
            *LeastCostStrategy().order([d for d in rest if d.id not in tagged], request, breaker),
        ]


def _tag_matches(candidates: list[Deployment], request: ChatRequest) -> list[Deployment]:
    wanted = set(request.tags)
    return [d for d in candidates if wanted & set(d.tags)] if wanted else []


_STRATEGIES: dict[str, RoutingStrategy] = {
    LeastCostStrategy.name: LeastCostStrategy(),
    LowestLatencyStrategy.name: LowestLatencyStrategy(),
    WeightedStrategy.name: WeightedStrategy(),
    PriorityStrategy.name: PriorityStrategy(),
    ConditionalStrategy.name: ConditionalStrategy(),
}

#: Accept a few friendlier spellings.
_ALIASES = {
    "cost": "least-cost",
    "cheapest": "least-cost",
    "latency": "lowest-latency",
    "fastest": "lowest-latency",
    "simple-shuffle": "weighted",
    "round-robin": "weighted",
}


def get_strategy(name: str | None) -> RoutingStrategy:
    if not name:
        return _STRATEGIES["priority"]
    resolved = _ALIASES.get(name, name)
    return _STRATEGIES.get(resolved, _STRATEGIES["priority"])


def list_strategies() -> list[str]:
    return sorted(_STRATEGIES)
