"""Routing strategies.

A strategy picks one deployment from the healthy candidates that can serve a
request. Strategies never perform I/O: they score what the registry and the
circuit breaker already know.
"""

from __future__ import annotations

import abc
import random
import re
from typing import TYPE_CHECKING

from app.core.schemas import ChatRequest, Role
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


#: Words that usually signal a task needing real reasoning, not a lookup.
_REASONING_TERMS = re.compile(
    r"\b(algorithm|analy[sz]e|architect\w*|compare|debug|derive|design|evaluate|"
    r"explain why|implement|optimi[sz]e|proof|prove|refactor|step[- ]by[- ]step|"
    r"trade-?offs?)\b",
    re.IGNORECASE,
)
_CODE_HINTS = re.compile(r"```|\bdef \w+\(|\bclass \w+|\bfunction \w+\(|=>|;\s*$", re.MULTILINE)
_LIST_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*])\s+", re.MULTILINE)


def score_complexity(request: ChatRequest) -> tuple[int, list[str]]:
    """Score how demanding a request looks, with the signals that fired.

    A cheap lexical heuristic, not a model call: length, code, reasoning
    vocabulary, multi-part asks, tool use, and conversation depth each add
    points. Wording is read from the latest user message only, so a long
    system prompt does not make every request look hard.
    """
    score = 0
    signals: list[str] = []

    prompt_tokens = _estimated_prompt_tokens(request)
    if prompt_tokens >= 1500:
        score += 2
        signals.append("long prompt")
    elif prompt_tokens >= 400:
        score += 1
        signals.append("medium prompt")

    users = [m for m in request.messages if m.role == Role.USER]
    latest = users[-1].text() if users else ""

    if _CODE_HINTS.search(latest):
        score += 2
        signals.append("code")

    terms = sorted({match.lower() for match in _REASONING_TERMS.findall(latest)})
    if terms:
        score += min(len(terms), 2)
        signals.append(f"reasoning terms: {', '.join(terms)}")

    if latest.count("?") >= 2 or len(_LIST_ITEM.findall(latest)) >= 3:
        score += 1
        signals.append("multi-part")

    if request.requires_tools():
        score += 1
        signals.append("tools")

    if len(request.non_system_messages()) > 6:
        score += 1
        signals.append("long conversation")

    return score, signals


class ComplexityStrategy(RoutingStrategy):
    """Send demanding prompts to frontier models and easy ones to cheap models.

    The default for `model: auto`. Request `tags` still win, as in the
    conditional strategy. Otherwise `score_complexity` decides the tier: at or
    above the threshold, the cheapest `frontier`-tagged deployment (the priciest
    one when nothing is tagged, price standing in for quality); below it, the
    cheapest `cheap`-tagged one (the cheapest overall when nothing is tagged).
    Deployments whose context window cannot hold the prompt go last.
    """

    name = "complexity"

    def __init__(self, threshold: int = 3) -> None:
        self._threshold = threshold

    def select(
        self, candidates: list[Deployment], request: ChatRequest, breaker: CircuitBreaker
    ) -> tuple[Deployment, str]:
        ordered, reason = self._plan(candidates, request)
        return ordered[0], reason

    def order(
        self, candidates: list[Deployment], request: ChatRequest, breaker: CircuitBreaker
    ) -> list[Deployment]:
        return self._plan(candidates, request)[0]

    def _plan(
        self, candidates: list[Deployment], request: ChatRequest
    ) -> tuple[list[Deployment], str]:
        prompt_tokens = _estimated_prompt_tokens(request)
        roomy = [d for d in candidates if d.capabilities.max_context_tokens >= prompt_tokens]
        pool = roomy or candidates
        cramped = [d for d in candidates if d not in pool]
        by_cost = sorted(
            pool,
            key=lambda d: (
                d.pricing.estimate(prompt_tokens, request.max_tokens or 512),
                d.priority,
            ),
        )

        tagged = _tag_matches(by_cost, request)
        if tagged:
            shared = sorted(set(tagged[0].tags) & set(request.tags))
            reason = f"request tags {', '.join(shared)} matched deployment tags"
            return _dedupe([*tagged, *by_cost, *cramped]), reason

        score, signals = score_complexity(request)
        hard = score >= self._threshold
        tier = "frontier" if hard else "cheap"
        # Hard work falls back toward stronger models, easy work toward cheaper ones.
        fallback = list(reversed(by_cost)) if hard else by_cost
        preferred = [d for d in by_cost if tier in d.tags] or fallback

        why = ", ".join(signals) if signals else "no complexity signals"
        reason = f"complexity {score}/{self._threshold} ({why}) -> {tier} tier"
        return _dedupe([*preferred, *fallback, *cramped]), reason


def _dedupe(deployments: list[Deployment]) -> list[Deployment]:
    seen: set[str] = set()
    unique: list[Deployment] = []
    for dep in deployments:
        if dep.id not in seen:
            seen.add(dep.id)
            unique.append(dep)
    return unique


def _tag_matches(candidates: list[Deployment], request: ChatRequest) -> list[Deployment]:
    wanted = set(request.tags)
    return [d for d in candidates if wanted & set(d.tags)] if wanted else []


_STRATEGIES: dict[str, RoutingStrategy] = {
    LeastCostStrategy.name: LeastCostStrategy(),
    LowestLatencyStrategy.name: LowestLatencyStrategy(),
    WeightedStrategy.name: WeightedStrategy(),
    PriorityStrategy.name: PriorityStrategy(),
    ConditionalStrategy.name: ConditionalStrategy(),
    ComplexityStrategy.name: ComplexityStrategy(),
}

#: Accept a few friendlier spellings.
_ALIASES = {
    "cost": "least-cost",
    "cheapest": "least-cost",
    "latency": "lowest-latency",
    "fastest": "lowest-latency",
    "simple-shuffle": "weighted",
    "round-robin": "weighted",
    "auto": "complexity",
}


def get_strategy(name: str | None) -> RoutingStrategy:
    if not name:
        return _STRATEGIES["priority"]
    resolved = _ALIASES.get(name, name)
    return _STRATEGIES.get(resolved, _STRATEGIES["priority"])


def list_strategies() -> list[str]:
    return sorted(_STRATEGIES)
