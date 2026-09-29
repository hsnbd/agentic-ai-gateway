from __future__ import annotations

import abc
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


class LlmJudgeRule(Rule):
    """Reserved interface for model-assisted policy judgments; not available yet."""

    def __init__(
        self,
        model: str,
        prompt: str,
        threshold: float,
        *,
        name: str = "llm_judge",
        action: Action = Action.FLAG,
        severity: Severity = Severity.MEDIUM,
    ) -> None:
        super().__init__(name, action, severity)
        self.model = model
        self.prompt = prompt
        self.threshold = threshold

    def evaluate(self, text: str) -> RuleMatch | None:
        raise NotImplementedError("LLM-judge guardrails are not enabled in this version.")
