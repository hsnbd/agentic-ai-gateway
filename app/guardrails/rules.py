from __future__ import annotations

import re
from collections.abc import Iterable

from app.guardrails.base import Action, Rule, RuleMatch, Severity, Span


class RegexRule(Rule):
    def __init__(
        self,
        name: str,
        pattern: str,
        action: Action,
        severity: Severity,
        *,
        ignorecase: bool = False,
        multiline: bool = False,
        replacement: str = "[REDACTED]",
    ) -> None:
        super().__init__(name, action, severity)
        flags = (re.IGNORECASE if ignorecase else 0) | (re.MULTILINE if multiline else 0)
        self.pattern = re.compile(pattern, flags)
        self.replacement = replacement

    def evaluate(self, text: str) -> RuleMatch | None:
        matches = list(self.pattern.finditer(text))
        if not matches:
            return None
        first = matches[0]
        return RuleMatch(
            rule_name=self.name,
            action=self.action,
            severity=self.severity,
            match_count=len(matches),
            excerpt=first.group(0),
            start=first.start(),
            end=first.end(),
        )

    def redaction_spans(self, text: str) -> list[Span]:
        return [
            Span(m.start(), m.end(), self.name, m.expand(self.replacement))
            for m in self.pattern.finditer(text)
        ]


class DenylistRule(Rule):
    def __init__(
        self,
        name: str,
        terms: Iterable[str],
        action: Action,
        severity: Severity,
        *,
        phrases: bool = False,
        replacement: str = "[REDACTED]",
    ) -> None:
        super().__init__(name, action, severity)
        normalized_terms = [term for term in terms if term]
        self.pattern: re.Pattern[str] | None = None
        if normalized_terms:
            alternatives = []
            for term in normalized_terms:
                escaped = re.escape(term)
                if phrases:
                    escaped = escaped.replace(r"\ ", r"\s+")
                alternatives.append(escaped)
            self.pattern = re.compile(rf"(?<!\w)(?:{'|'.join(alternatives)})(?!\w)", re.IGNORECASE)
        self.replacement = replacement

    def evaluate(self, text: str) -> RuleMatch | None:
        if self.pattern is None:
            return None
        matches = list(self.pattern.finditer(text))
        if not matches:
            return None
        first = matches[0]
        return RuleMatch(
            rule_name=self.name,
            action=self.action,
            severity=self.severity,
            match_count=len(matches),
            excerpt=first.group(0),
            start=first.start(),
            end=first.end(),
        )

    def redaction_spans(self, text: str) -> list[Span]:
        if self.pattern is None:
            return []
        return [
            Span(m.start(), m.end(), self.name, self.replacement)
            for m in self.pattern.finditer(text)
        ]


class LengthRule(Rule):
    def __init__(
        self,
        name: str,
        action: Action,
        severity: Severity,
        *,
        max_chars: int | None = None,
        max_messages: int | None = None,
    ) -> None:
        super().__init__(name, action, severity)
        if max_chars is None and max_messages is None:
            raise ValueError("LengthRule requires max_chars and/or max_messages")
        if max_chars is not None and max_chars < 0:
            raise ValueError("max_chars must be non-negative")
        if max_messages is not None and max_messages < 0:
            raise ValueError("max_messages must be non-negative")
        self.max_chars = max_chars
        self.max_messages = max_messages

    def evaluate(self, text: str, *, message_count: int | None = None) -> RuleMatch | None:
        exceeded: dict[str, int] = {}
        if self.max_chars is not None and len(text) > self.max_chars:
            exceeded["characters"] = len(text)
        if (
            self.max_messages is not None
            and message_count is not None
            and message_count > self.max_messages
        ):
            exceeded["messages"] = message_count
        if not exceeded:
            return None
        return RuleMatch(
            rule_name=self.name,
            action=self.action,
            severity=self.severity,
            match_count=1,
            excerpt=text[:200],
            details={
                "exceeded": exceeded,
                "max_chars": self.max_chars,
                "max_messages": self.max_messages,
            },
        )


class TopicRule(Rule):
    """Keyword pre-filter: flags text containing at least min_hits distinct keywords."""

    def __init__(
        self,
        name: str,
        keywords: Iterable[str],
        action: Action,
        severity: Severity,
        *,
        min_hits: int = 1,
    ) -> None:
        super().__init__(name, action, severity)
        if min_hits < 1:
            raise ValueError("min_hits must be at least 1")
        self.keywords = frozenset(keyword.casefold() for keyword in keywords if keyword)
        self.min_hits = min_hits

    def evaluate(self, text: str) -> RuleMatch | None:
        lowered = text.casefold()
        hits = sorted(keyword for keyword in self.keywords if keyword in lowered)
        if len(hits) < self.min_hits:
            return None
        return RuleMatch(
            rule_name=self.name,
            action=self.action,
            severity=self.severity,
            match_count=len(hits),
            excerpt=", ".join(hits),
            details={"keywords": hits},
        )
