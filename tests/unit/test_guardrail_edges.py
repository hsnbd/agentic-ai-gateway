"""Configuration, rule, and stage edge cases for guardrails."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.errors import ConfigurationError, GuardrailViolationError
from app.core.schemas import ChatRequest, ChatResponse, Choice, Message, Role
from app.guardrails.base import (
    Action,
    GuardrailResult,
    LlmJudgeRule,
    Phase,
    Rule,
    RuleMatch,
    Severity,
    _mask_excerpt,
    _parse_verdict,
)
from app.guardrails.pii import PiiRule, _passes_luhn
from app.guardrails.registry import GuardrailRegistry, Policy
from app.guardrails.rules import DenylistRule, LengthRule, RegexRule, TopicRule
from app.guardrails.stage import InputGuardrailStage, OutputGuardrailStage, _redact_content
from tests.unit.test_guardrails import _context, _load_registry

# -- Base helpers -----------------------------------------------------------


def test_excerpts_are_masked_by_length() -> None:
    assert _mask_excerpt("") == ""
    assert _mask_excerpt("secret") == "se***et"
    assert _mask_excerpt("a-much-longer-secret") == "a-mu…cret"


def test_result_summary_counts_actions() -> None:
    result = GuardrailResult(policy="p", phase=Phase.INPUT)
    assert result.summary() == "input guardrails passed (p)"
    result.matches = [
        RuleMatch("a", Action.BLOCK, Severity.HIGH, 1, "x"),
        RuleMatch("b", Action.FLAG, Severity.LOW, 1, "y"),
        RuleMatch("c", Action.FLAG, Severity.LOW, 1, "z"),
    ]
    assert result.summary() == "input guardrails block=1, flag=2 (p)"


def test_base_rule_is_non_redacting() -> None:
    class Plain(Rule):
        def evaluate(self, text: str) -> RuleMatch | None:
            return None

    rule = Plain("plain", Action.REDACT, Severity.LOW)
    assert rule.redaction_spans("anything") == []
    assert rule.redact("anything") == "anything"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [({"threshold": 1.5}, "threshold"), ({"threshold": 0.5, "on_error": "retry"}, "on_error")],
)
def test_llm_judge_validates_configuration(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LlmJudgeRule("m", "p", **kwargs)


async def test_llm_judge_skips_blank_text_and_sync_evaluate() -> None:
    rule = LlmJudgeRule("m", "p", 0.5)
    assert rule.evaluate("anything") is None
    assert await rule.aevaluate("   ") is None
    # No judge configured and failing open.
    assert await rule.aevaluate("text") is None


@pytest.mark.parametrize(
    ("reply", "message"),
    [("no json here", "not JSON"), ('{"score": 3}', "outside 0..1")],
)
def test_parse_verdict_rejects_bad_replies(reply: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _parse_verdict(reply)


def test_parse_verdict_extracts_json_from_prose() -> None:
    assert _parse_verdict('Verdict: {"score": 0.25} done') == (0.25, "")


# -- Rules ------------------------------------------------------------------


def test_regex_rule_flags_and_substitutes_groups() -> None:
    rule = RegexRule(
        "key",
        r"key=(\w+)",
        Action.REDACT,
        Severity.HIGH,
        ignorecase=True,
        multiline=True,
        replacement=r"key=[\1]",
    )
    assert rule.redact("KEY=abc and key=def") == "key=[abc] and key=[def]"
    assert rule.evaluate("nothing") is None


def test_empty_denylist_never_matches() -> None:
    rule = DenylistRule("empty", ["", ""], Action.REDACT, Severity.LOW)
    assert rule.evaluate("anything") is None
    assert rule.redaction_spans("anything") == []


def test_denylist_phrases_match_flexible_whitespace() -> None:
    rule = DenylistRule("phrase", ["top secret"], Action.REDACT, Severity.LOW, phrases=True)
    assert rule.redact("this is top   secret stuff") == "this is [REDACTED] stuff"
    assert rule.evaluate("nothing") is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "requires max_chars"),
        ({"max_chars": -1}, "max_chars"),
        ({"max_messages": -1}, "max_messages"),
    ],
)
def test_length_rule_validates_limits(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        LengthRule("len", Action.BLOCK, Severity.LOW, **kwargs)


def test_length_rule_counts_messages() -> None:
    rule = LengthRule("len", Action.BLOCK, Severity.LOW, max_messages=2)
    match = rule.evaluate("x", message_count=3)
    assert match is not None and match.details["exceeded"] == {"messages": 3}
    assert rule.evaluate("x") is None
    assert rule.evaluate("x", message_count=2) is None


def test_topic_rule_requires_minimum_hits() -> None:
    with pytest.raises(ValueError, match="min_hits"):
        TopicRule("t", ["a"], Action.FLAG, Severity.LOW, min_hits=0)
    rule = TopicRule("t", ["Weapons", "explosives", ""], Action.FLAG, Severity.LOW, min_hits=2)
    assert rule.evaluate("weapons only") is None
    match = rule.evaluate("WEAPONS and explosives")
    assert match is not None and match.details["keywords"] == ["explosives", "weapons"]


def test_pii_rule_validates_entities_and_mode() -> None:
    with pytest.raises(ValueError, match="Unsupported PII entity"):
        PiiRule("pii", ["DNA"], Action.REDACT, Severity.HIGH)
    with pytest.raises(ValueError, match="mode"):
        PiiRule("pii", ["EMAIL"], Action.REDACT, Severity.HIGH, mode="hash")


def test_luhn_rejects_wrong_lengths() -> None:
    assert not _passes_luhn("4242")
    assert not _passes_luhn("4" * 20)


# -- Registry configuration -------------------------------------------------


def test_registry_requires_default_policy() -> None:
    with pytest.raises(ConfigurationError, match="'default' policy"):
        GuardrailRegistry({"strict": Policy(name="strict")})


def test_registry_load_errors(tmp_path: Any) -> None:
    with pytest.raises(ConfigurationError, match="Could not load"):
        GuardrailRegistry.load(tmp_path / "missing.yaml")
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("policies: [", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="Could not load"):
        GuardrailRegistry.load(bad_yaml)
    no_policies = tmp_path / "none.yaml"
    no_policies.write_text("other: 1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="policies mapping"):
        GuardrailRegistry.load(no_policies)


@pytest.mark.parametrize(
    ("policy_yaml", "message"),
    [
        ("  default: 5\n", "must have a name and mapping"),
        ("  default:\n    input: {a: 1}\n", "must be a list"),
        ("  default:\n    input: [5]\n", "must be a mapping"),
        ("  default:\n    input: [{type: regex}]\n", "requires a name"),
        (
            "  default:\n    input: [{type: regex, name: r, pattern: x, action: explode}]\n",
            "Invalid action or severity",
        ),
        (
            "  default:\n    input: [{type: regex, name: r, pattern: '('}]\n",
            "Invalid configuration",
        ),
        (
            "  default:\n    input: [{type: denylist, name: d, terms: [1, 2]}]\n",
            "list of strings",
        ),
    ],
)
def test_invalid_rule_configuration(tmp_path: Any, policy_yaml: str, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        _load_registry(tmp_path, policy_yaml)


def test_every_rule_type_loads(tmp_path: Any) -> None:
    registry = _load_registry(
        tmp_path,
        """  default:
    description: everything
    input:
      - {type: regex, name: r, pattern: 'x+'}
      - {type: denylist, name: d, terms: [bad], phrases: true}
      - {type: length, name: l, max_chars: 10}
      - {type: topic, name: t, keywords: [a], min_hits: 1}
      - {type: pii, name: p, entities: [EMAIL], mode: mask}
      - {type: llm_judge, name: j, model: m, prompt: p, threshold: 0.7, on_error: block}
    output: []
""",
    )
    rules = registry.get_policy("default").input_rules
    assert [type(rule).__name__ for rule in rules] == [
        "RegexRule",
        "DenylistRule",
        "LengthRule",
        "TopicRule",
        "PiiRule",
        "LlmJudgeRule",
    ]
    assert registry.list_policies() == ["default"]


# -- Stages -----------------------------------------------------------------

REDACTING = """  default:
    input:
      - {type: regex, name: secret, pattern: 'sk-\\w+', action: redact}
    output:
      - {type: regex, name: secret, pattern: 'sk-\\w+', action: redact}
      - {type: regex, name: forbidden, pattern: 'FORBIDDEN', action: block}
"""


def _request(*messages: Message) -> ChatRequest:
    return ChatRequest(model="m", messages=list(messages))


async def test_input_stage_redacts_user_and_system_text(tmp_path: Any) -> None:
    registry = _load_registry(tmp_path, REDACTING)
    system = Message(role=Role.SYSTEM, content="use sk-system")
    user = Message(role=Role.USER, content="my key sk-user")
    assistant = Message(role=Role.ASSISTANT, content="sk-left-alone")
    ctx = _context(_request(system, user, assistant))
    await InputGuardrailStage(registry).process(ctx)
    assert system.content == "use [REDACTED]"
    assert user.content == "my key [REDACTED]"
    assert assistant.content == "sk-left-alone"
    assert ctx.guardrail_results["input"]["redacted"] is True


async def test_output_stage_redacts_and_blocks(tmp_path: Any) -> None:
    registry = _load_registry(tmp_path, REDACTING)
    stage = OutputGuardrailStage(registry)
    response = ChatResponse(
        model="m", choices=[Choice(message=Message(role=Role.ASSISTANT, content="sk-out"))]
    )
    ctx = _context(_request(Message(role=Role.USER, content="hi")))
    redacted = await stage.finalize(ctx, response)
    assert redacted.choices[0].message.content == "[REDACTED]"

    blocked = ChatResponse(
        model="m", choices=[Choice(message=Message(role=Role.ASSISTANT, content="FORBIDDEN"))]
    )
    with pytest.raises(GuardrailViolationError, match="forbidden"):
        await stage.finalize(ctx, blocked)


async def test_output_stage_record_and_stream_redactor(tmp_path: Any) -> None:
    registry = _load_registry(tmp_path, REDACTING)
    stage = OutputGuardrailStage(registry, stream_holdback=16)
    ctx = _context(_request(Message(role=Role.USER, content="hi")))
    assert stage.stream_redactor(ctx) is not None
    await stage.record(ctx, "partial FORBIDDEN text")
    assert ctx.guardrail_results["output"]["blocked"] is True

    passive = _load_registry(tmp_path, "  default:\n    output: []\n")
    assert OutputGuardrailStage(passive).stream_redactor(ctx) is None


def test_redact_content_leaves_none_and_images_untouched() -> None:
    from app.core.schemas import ImagePart, TextPart

    assert _redact_content(None, str.upper) is None
    image = ImagePart(url="https://example.com/a.png")
    parts = _redact_content([TextPart(text="a"), image], str.upper)
    assert parts == [TextPart(text="A"), image]


async def test_input_stage_blocks_and_counts_messages(tmp_path: Any) -> None:
    registry = _load_registry(
        tmp_path,
        """  default:
    input:
      - {type: length, name: too_many, max_messages: 1, action: block}
""",
    )
    ctx = _context(
        _request(Message(role=Role.USER, content="a"), Message(role=Role.USER, content="b"))
    )
    with pytest.raises(GuardrailViolationError, match="too_many") as raised:
        await InputGuardrailStage(registry).process(ctx)
    assert raised.value.details == {"rule": "too_many", "policy": "default", "phase": "input"}


def test_set_judge_skips_non_judge_rules(tmp_path: Any) -> None:
    registry = _load_registry(
        tmp_path,
        """  default:
    input:
      - {type: regex, name: r, pattern: x}
      - {type: llm_judge, name: j, model: m, prompt: p}
""",
    )

    async def judge(model: str, instructions: str, text: str) -> str:
        return '{"score": 0}'

    registry.set_judge(judge)
    regex, judge_rule = registry.get_policy("default").input_rules
    assert not hasattr(regex, "judge")
    assert judge_rule.judge is judge  # type: ignore[attr-defined]


async def test_violations_are_persisted_and_clean_text_is_not(tmp_path: Any) -> None:
    from types import SimpleNamespace

    from sqlalchemy import select

    from app.config.settings import Settings
    from app.core.pipeline import RequestContext
    from app.db.models import GuardrailViolation
    from app.db.session import Database

    database = Database(Settings(database_url="sqlite+aiosqlite:///:memory:"))
    await database.startup()
    await database.create_all()
    try:
        registry = _load_registry(tmp_path, REDACTING)
        stage = InputGuardrailStage(registry)
        state = SimpleNamespace(db=database)
        clean = RequestContext(
            request=_request(Message(role=Role.USER, content="hello")),
            state=state,  # type: ignore[arg-type]
        )
        await stage.process(clean)
        dirty = RequestContext(
            request=_request(Message(role=Role.USER, content="sk-leak")),
            state=state,  # type: ignore[arg-type]
            key_id="k1",
        )
        await stage.process(dirty)
        async with database.session() as session:
            rows = list((await session.scalars(select(GuardrailViolation))).all())
        assert [(row.rule, row.virtual_key_id, row.phase) for row in rows] == [
            ("secret", "k1", "input")
        ]
    finally:
        await database.shutdown()


async def test_output_stage_passes_clean_responses_through(tmp_path: Any) -> None:
    registry = _load_registry(tmp_path, REDACTING)
    response = ChatResponse(
        model="m", choices=[Choice(message=Message(role=Role.ASSISTANT, content="all fine"))]
    )
    ctx = _context(_request(Message(role=Role.USER, content="hi")))
    assert await OutputGuardrailStage(registry).finalize(ctx, response) is response
    assert response.choices[0].message.content == "all fine"
