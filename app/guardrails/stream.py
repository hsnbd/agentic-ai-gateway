"""Output guardrails for streamed responses.

Text can't be recalled once it reaches the client, so a redaction that is
only found after the whole stream has passed through comes too late. The
redactor holds back the most recent `holdback` characters: text is released
only once no redaction rule could still extend a match into it, and a match
that straddles the release point holds everything from its start.

A BLOCK rule that fires ends the stream with a guardrail error. Anything
already released stays released, which is why the holdback exists.
"""

from __future__ import annotations

from app.core.errors import GuardrailViolationError
from app.guardrails.base import Action, Phase, Span, apply_spans, resolve_overlaps
from app.guardrails.registry import GuardrailRegistry
from app.guardrails.rules import LengthRule


class StreamRedactor:
    def __init__(self, registry: GuardrailRegistry, policy_name: str, holdback: int) -> None:
        policy = registry.get_policy(policy_name)
        self.policy = policy.name
        self.rules = [
            rule for rule in policy.rules_for(Phase.OUTPUT) if not isinstance(rule, LengthRule)
        ]
        self.holdback = max(0, holdback)
        self._raw = ""
        #: Offset into `_raw` up to which text has been released.
        self._released = 0

    @property
    def active(self) -> bool:
        return bool(self.rules)

    def feed(self, text: str) -> str:
        """Add streamed text; return whatever is now safe to send."""
        self._raw += text
        return self._release(final=False)

    def finish(self) -> str:
        """Release everything still held once the stream ends."""
        return self._release(final=True)

    def _release(self, *, final: bool) -> str:
        # Matches are bounded by the holdback, so scanning from one holdback
        # before the release point sees every match that can still change.
        base = max(0, self._released - self.holdback)
        window = self._raw[base:]
        self._check_block(window)

        spans: list[Span] = []
        for rule in self.rules:
            if rule.action == Action.REDACT:
                spans.extend(
                    Span(s.start + base, s.end + base, s.label, s.replacement, s.priority)
                    for s in rule.redaction_spans(window)
                )
        spans = resolve_overlaps(spans)

        end = len(self._raw) if final else max(self._released, len(self._raw) - self.holdback)
        for span in spans:
            if span.start < end < span.end:
                end = max(self._released, span.start)
        if end <= self._released:
            return ""

        start = self._released
        segment_spans = [
            Span(
                max(span.start, start) - start,
                min(span.end, end) - start,
                span.label,
                span.replacement,
                span.priority,
            )
            for span in spans
            if span.end > start and span.start < end
        ]
        self._released = end
        return apply_spans(self._raw[start:end], segment_spans)

    def _check_block(self, text: str) -> None:
        for rule in self.rules:
            if rule.action != Action.BLOCK:
                continue
            if rule.evaluate(text) is not None:
                raise GuardrailViolationError(
                    f"Output blocked by guardrail rule {rule.name!r}",
                    details={"rule": rule.name, "policy": self.policy, "phase": "output"},
                )
