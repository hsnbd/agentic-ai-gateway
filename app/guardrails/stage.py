from __future__ import annotations

import logging
from collections.abc import Callable

from app.core.errors import GuardrailViolationError
from app.core.pipeline import RequestContext
from app.core.schemas import ChatResponse, ContentPart, Role, TextPart
from app.db.models import GuardrailViolation
from app.guardrails.base import Action, GuardrailResult, Phase
from app.guardrails.registry import GuardrailRegistry
from app.guardrails.stream import StreamRedactor

logger = logging.getLogger(__name__)


class InputGuardrailStage:
    name = "guardrails_input"

    def __init__(self, registry: GuardrailRegistry) -> None:
        self.registry = registry

    async def process(self, ctx: RequestContext) -> None:
        policy = ctx.request.guardrail_policy or "default"
        messages = [
            message
            for message in ctx.request.messages
            if message.role in {Role.USER, Role.SYSTEM}
        ]
        text = "\n".join(message.text() for message in messages)
        result = await self.registry.evaluate(
            policy, text, Phase.INPUT, message_count=len(messages)
        )
        _record_result(ctx, "input", result)
        ctx.guardrail_flagged = ctx.guardrail_flagged or result.flagged
        await _persist_violations(ctx, result)
        if result.blocked:
            blocked = next(match for match in result.matches if match.action == Action.BLOCK)
            raise GuardrailViolationError(
                f"Input blocked by guardrail rule {blocked.rule_name!r}",
                details={
                    "rule": blocked.rule_name,
                    "policy": result.policy,
                    "phase": Phase.INPUT.value,
                },
            )
        if result.redacted:
            for message in messages:
                message.content = _redact_content(
                    message.content,
                    lambda value: self.registry.redact_text(policy, value, Phase.INPUT),
                )
        return None


class OutputGuardrailStage:
    name = "guardrails_output"

    def __init__(self, registry: GuardrailRegistry, stream_holdback: int = 128) -> None:
        self.registry = registry
        self.stream_holdback = stream_holdback

    def stream_redactor(self, ctx: RequestContext) -> StreamRedactor | None:
        """A filter that applies this policy to text while it streams."""
        redactor = StreamRedactor(
            self.registry, ctx.request.guardrail_policy or "default", self.stream_holdback
        )
        return redactor if redactor.active else None

    async def record(self, ctx: RequestContext, text: str) -> None:
        """Evaluate and persist violations without acting on them (stream already cut)."""
        policy = ctx.request.guardrail_policy or "default"
        result = await self.registry.evaluate(policy, text, Phase.OUTPUT)
        _record_result(ctx, "output", result)
        ctx.guardrail_flagged = ctx.guardrail_flagged or result.flagged
        await _persist_violations(ctx, result)

    async def finalize(self, ctx: RequestContext, response: ChatResponse) -> ChatResponse:
        policy = ctx.request.guardrail_policy or "default"
        result = await self.registry.evaluate(policy, response.text, Phase.OUTPUT)
        _record_result(ctx, "output", result)
        ctx.guardrail_flagged = ctx.guardrail_flagged or result.flagged
        await _persist_violations(ctx, result)
        if result.blocked:
            blocked = next(match for match in result.matches if match.action == Action.BLOCK)
            raise GuardrailViolationError(
                f"Output blocked by guardrail rule {blocked.rule_name!r}",
                details={
                    "rule": blocked.rule_name,
                    "policy": result.policy,
                    "phase": Phase.OUTPUT.value,
                },
            )
        if result.redacted and response.choices:
            message = response.choices[0].message
            message.content = _redact_content(
                message.content,
                lambda value: self.registry.redact_text(policy, value, Phase.OUTPUT),
            )
        return response


def _redact_content(
    content: str | list[ContentPart] | None,
    redact: Callable[[str], str],
) -> str | list[ContentPart] | None:
    if isinstance(content, str):
        return redact(content)
    if isinstance(content, list):
        return [
            TextPart(text=redact(part.text)) if isinstance(part, TextPart) else part
            for part in content
        ]
    return content


def _record_result(ctx: RequestContext, key: str, result: GuardrailResult) -> None:
    existing = ctx.guardrail_results.get(key)
    record = existing if isinstance(existing, dict) else {}
    record.update(result.to_dict())
    ctx.guardrail_results[key] = record


async def _persist_violations(ctx: RequestContext, result: GuardrailResult) -> None:
    if not result.matches:
        return
    try:
        async with ctx.state.db.session() as session:
            for match in result.matches:
                session.add(
                    GuardrailViolation(
                        request_id=ctx.request_id,
                        virtual_key_id=ctx.key_id,
                        policy=result.policy,
                        rule=match.rule_name,
                        phase=result.phase.value,
                        action=match.action.value,
                        severity=match.severity.value,
                        match_count=match.match_count,
                        excerpt=match.excerpt,
                        details=match.details,
                    )
                )
    except Exception:
        logger.exception("Could not persist guardrail violations for request %s", ctx.request_id)
