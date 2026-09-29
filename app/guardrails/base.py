from __future__ import annotations

import abc
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Action(StrEnum):
    BLOCK = "block"
    REDACT = "redact"
    FLAG = "flag"
    ALLOW = "allow"


class Severity(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Phase(StrEnum):
    INPUT = "input"
    OUTPUT = "output"


def _mask_excerpt(value: str) -> str:
    truncated = value[:200]
    if not truncated:
        return truncated
    if len(truncated) <= 8:
        return f"{truncated[:2]}***{truncated[-2:]}"
    return f"{truncated[:4]}…{truncated[-4:]}"


@dataclass
class RuleMatch:
    rule_name: str
    action: Action
    severity: Severity
    match_count: int
    excerpt: str
    start: int | None = None
    end: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.excerpt = _mask_excerpt(self.excerpt)


@dataclass
class GuardrailResult:
    policy: str
    phase: Phase
    matches: list[RuleMatch] = field(default_factory=list)
    text: str = ""
    blocked: bool = False
    flagged: bool = False
    redacted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "phase": self.phase.value,
            "matches": [
                {
                    "rule_name": match.rule_name,
                    "action": match.action.value,
                    "severity": match.severity.value,
                    "match_count": match.match_count,
                    "excerpt": match.excerpt,
                    "start": match.start,
                    "end": match.end,
                    "details": match.details,
                }
                for match in self.matches
            ],
            "blocked": self.blocked,
            "flagged": self.flagged,
            "redacted": self.redacted,
        }

    def summary(self) -> str:
        if not self.matches:
            return f"{self.phase.value} guardrails passed ({self.policy})"
        actions = ", ".join(
            f"{action}={sum(match.action.value == action for match in self.matches)}"
            for action in ("block", "redact", "flag", "allow")
            if any(match.action.value == action for match in self.matches)
        )
        return f"{self.phase.value} guardrails {actions} ({self.policy})"


@dataclass(frozen=True)
class Span:
    """A detected region of the ORIGINAL text and what should replace it."""

    start: int
    end: int
    label: str
    replacement: str
    #: Lower wins when two candidates start at the same offset with the same length.
    priority: int = 100

    @property
    def length(self) -> int:
        return self.end - self.start


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Pick a deterministic, non-overlapping subset of candidate spans.

    Tie-break: sort by (start, -length, priority, label) and greedily keep each span that
    begins at or after the end of the last kept span. The earliest-starting span therefore
    always wins, so a card candidate that begins inside a phone number is discarded instead
    of swallowing the phone's tail. Among spans with the same start, the longest wins; among
    equal-length spans, the more specific detector (lower priority value) wins.
    """
    ordered = sorted(spans, key=lambda s: (s.start, -s.length, s.priority, s.label))
    kept: list[Span] = []
    last_end = -1
    for span in ordered:
        if span.length > 0 and span.start >= last_end:
            kept.append(span)
            last_end = span.end
    return kept


def apply_spans(text: str, spans: list[Span]) -> str:
    """Substitute resolved spans in one pass over the original text.

    Spans are applied in reverse start order so each replacement leaves the offsets of
    every earlier span untouched.
    """
    result = text
    for span in sorted(resolve_overlaps(spans), key=lambda s: s.start, reverse=True):
        result = result[: span.start] + span.replacement + result[span.end :]
    return result


class Rule(abc.ABC):
    def __init__(self, name: str, action: Action, severity: Severity) -> None:
        self.name = name
        self.action = action
        self.severity = severity

    @abc.abstractmethod
    def evaluate(self, text: str) -> RuleMatch | None:
        raise NotImplementedError

    def redaction_spans(self, text: str) -> list[Span]:
        """Spans (over ``text``) this rule would replace; empty for non-redacting rules."""
        return []

    def redact(self, text: str) -> str:
        return apply_spans(text, self.redaction_spans(text))


#: Sends (model, instructions, text) to a model and returns its raw reply.
JudgeClient = Callable[[str, str, str], Awaitable[str]]

_JUDGE_INSTRUCTIONS = (
    "You are a guardrail judge for an AI gateway. Decide how strongly the text "
    "below violates this policy: {policy}\n"
    'Reply with JSON only, e.g. {{"score": 0.0, "reason": "..."}}, where score is '
    "0 (no violation) to 1 (clear violation)."
)


class LlmJudgeRule(Rule):
    """Ask a model whether text violates a natural-language policy.

    The judge is called through the gateway's provider registry (not the chat
    pipeline, so it is not itself guarded, cached, or billed to the caller).
    A score at or above `threshold` is a match. If the judge fails or replies
    with something unparseable, `on_error` decides: "allow" (default, fail
    open) or "block" (fail closed, which applies the rule's action).
    """

    is_async = True

    def __init__(
        self,
        model: str,
        prompt: str,
        threshold: float,
        *,
        name: str = "llm_judge",
        action: Action = Action.FLAG,
        severity: Severity = Severity.MEDIUM,
        on_error: str = "allow",
    ) -> None:
        super().__init__(name, action, severity)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between 0 and 1")
        if on_error not in {"allow", "block"}:
            raise ValueError("on_error must be 'allow' or 'block'")
        self.model = model
        self.prompt = prompt
        self.threshold = threshold
        self.on_error = on_error
        self.judge: JudgeClient | None = None

    def evaluate(self, text: str) -> RuleMatch | None:
        # Judging needs a model call; the registry awaits `aevaluate` instead.
        return None

    async def aevaluate(self, text: str) -> RuleMatch | None:
        if not text.strip():
            return None
        try:
            if self.judge is None:
                raise RuntimeError("no judge client is configured")
            reply = await self.judge(
                self.model, _JUDGE_INSTRUCTIONS.format(policy=self.prompt), text
            )
            score, reason = _parse_verdict(reply)
        except Exception as exc:
            if self.on_error == "block":
                return self._match(text, 1.0, f"judge unavailable: {exc}")
            return None
        return self._match(text, score, reason) if score >= self.threshold else None

    def _match(self, text: str, score: float, reason: str) -> RuleMatch:
        return RuleMatch(
            rule_name=self.name,
            action=self.action,
            severity=self.severity,
            match_count=1,
            excerpt=text[:80],
            details={"score": score, "reason": reason, "model": self.model},
        )


def _parse_verdict(reply: str) -> tuple[float, str]:
    """Pull `{"score": ..., "reason": ...}` out of a reply that may wrap it in prose."""
    start, end = reply.find("{"), reply.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"judge reply is not JSON: {reply[:120]!r}")
    verdict = json.loads(reply[start : end + 1])
    score = float(verdict["score"])
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"judge score {score} is outside 0..1")
    return score, str(verdict.get("reason", ""))
