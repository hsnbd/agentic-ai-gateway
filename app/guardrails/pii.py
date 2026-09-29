from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from app.guardrails.base import Action, Rule, RuleMatch, Severity, Span, resolve_overlaps

# Card candidates are captured inside a lookahead so every start offset yields a candidate;
# overlapping candidates are later resolved against all other detectors. The boundaries
# `(?<![\d-])` / `(?![\d-])` stop a run from starting or ending inside a hyphenated number
# (e.g. the "2671" tail of "415-555-2671"), and the Luhn check drops most remaining noise.
_CARD_PATTERN = re.compile(r"(?<![\d-])(?=((?:\d[ -]?){12,18}\d)(?![\d-]))")

_ENTITY_PATTERNS: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)+(?![\w.-])", re.I),
    "PHONE": re.compile(
        r"(?<![\w+])(?=(?:\D*\d){10,15}(?!\d))"
        r"(?:\+\d{10,15}|(?:\+\d{1,3}[\s.-]?)?"
        r"(?:\(\d{2,4}\)|\d{2,4})[\s.-]\d{3,4}[\s.-]\d{4})"
        r"(?![\w-])"
    ),
    "SSN": re.compile(
        r"(?<![\d-])(?!000|666|9\d\d)\d{3}[- ](?!00)\d{2}[- ](?!0000)\d{4}(?![\d-])"
    ),
    "CREDIT_CARD": _CARD_PATTERN,
    "IP_ADDRESS": re.compile(
        r"(?<![\d.])(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
        r"(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}(?![\d.])"
    ),
    "AWS_ACCESS_KEY": re.compile(r"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])"),
    "API_KEY": re.compile(
        r"(?i)(?<![\w])(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|"
        r"Bearer\s+[A-Za-z0-9._~+/=-]{12,}|"
        r"(?:api[_-]?key|secret)[\s:=]+[A-Za-z0-9._-]{8,})"
    ),
}

#: Detector specificity used only to break exact (start, length) ties; lower wins.
_PRIORITY: dict[str, int] = {
    "AWS_ACCESS_KEY": 0,
    "API_KEY": 1,
    "EMAIL": 2,
    "SSN": 3,
    "CREDIT_CARD": 4,
    "IP_ADDRESS": 5,
    "PHONE": 6,
}
_MASKABLE = frozenset({"CREDIT_CARD", "PHONE"})


class PiiRule(Rule):
    def __init__(
        self,
        name: str,
        entities: Iterable[str],
        action: Action,
        severity: Severity,
        *,
        mode: str = "full",
    ) -> None:
        super().__init__(name, action, severity)
        normalized = tuple(dict.fromkeys(entity.upper() for entity in entities))
        unknown = sorted(set(normalized) - _ENTITY_PATTERNS.keys())
        if unknown:
            raise ValueError(f"Unsupported PII entity type(s): {', '.join(unknown)}")
        if mode not in {"full", "mask"}:
            raise ValueError("PII mode must be 'full' or 'mask'")
        self.entities = normalized
        self.mode = mode

    def detect(self, text: str) -> list[Span]:
        """All configured entities as resolved, non-overlapping spans over ``text``."""
        candidates: list[Span] = []
        for entity in self.entities:
            for start, end in _candidates(text, entity):
                candidates.append(
                    Span(
                        start,
                        end,
                        entity,
                        self._replacement(entity, text[start:end]),
                        _PRIORITY[entity],
                    )
                )
        return resolve_overlaps(candidates)

    def evaluate(self, text: str) -> RuleMatch | None:
        spans = self.detect(text)
        if not spans:
            return None
        first = spans[0]
        return RuleMatch(
            rule_name=self.name,
            action=self.action,
            severity=self.severity,
            match_count=len(spans),
            excerpt=text[first.start : first.end],
            start=first.start,
            end=first.end,
            details={
                "entities": dict(sorted(Counter(s.label for s in spans).items())),
                "spans": [{"entity": s.label, "start": s.start, "end": s.end} for s in spans],
            },
        )

    def redaction_spans(self, text: str) -> list[Span]:
        return self.detect(text)

    def _replacement(self, entity: str, value: str) -> str:
        if self.mode == "mask" and entity in _MASKABLE:
            digits = re.sub(r"\D", "", value)
            return f"[{entity}_REDACTED:{'*' * max(len(digits) - 4, 0)}{digits[-4:]}]"
        return f"[{entity}_REDACTED]"


def _candidates(text: str, entity: str) -> list[tuple[int, int]]:
    pattern = _ENTITY_PATTERNS[entity]
    if entity == "CREDIT_CARD":
        return [
            (m.start(1), m.end(1)) for m in pattern.finditer(text) if _passes_luhn(m.group(1))
        ]
    return [m.span() for m in pattern.finditer(text)]


def _passes_luhn(value: str) -> bool:
    digits = [int(char) for char in value if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0
