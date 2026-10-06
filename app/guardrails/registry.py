from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from app.core.errors import ConfigurationError
from app.guardrails.base import (
    Action,
    GuardrailResult,
    JudgeClient,
    LlmJudgeRule,
    Phase,
    Rule,
    Severity,
    Span,
    apply_spans,
)
from app.guardrails.pii import PiiRule
from app.guardrails.rules import DenylistRule, LengthRule, RegexRule, TopicRule

logger = logging.getLogger(__name__)
_SUPPORTED_RULE_TYPES = {"regex", "denylist", "length", "topic", "pii", "llm_judge"}


@dataclass
class Policy:
    name: str
    description: str = ""
    input_rules: list[Rule] = field(default_factory=list)
    output_rules: list[Rule] = field(default_factory=list)
    #: Also screen MCP tool arguments (input rules) and results (output rules).
    apply_to_tools: bool = False

    def rules_for(self, phase: Phase) -> list[Rule]:
        return self.input_rules if phase == Phase.INPUT else self.output_rules


class GuardrailRegistry:
    def __init__(self, policies: dict[str, Policy]) -> None:
        if "default" not in policies:
            raise ConfigurationError("Guardrail configuration must define a 'default' policy")
        self._policies = policies

    @classmethod
    def load(cls, path: str | Path) -> GuardrailRegistry:
        config_path = Path(path)
        try:
            raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ConfigurationError(
                f"Could not load guardrail configuration from {config_path}: {exc}"
            ) from exc
        if not isinstance(raw, dict) or not isinstance(raw.get("policies"), dict):
            raise ConfigurationError("Guardrail configuration must contain a policies mapping")

        policies: dict[str, Policy] = {}
        for name, value in raw["policies"].items():
            if not isinstance(name, str) or not isinstance(value, dict):
                raise ConfigurationError("Each guardrail policy must have a name and mapping")
            policies[name] = Policy(
                name=name,
                description=str(value.get("description", "")),
                input_rules=_build_rules(value.get("input") or [], name, "input"),
                output_rules=_build_rules(value.get("output") or [], name, "output"),
                apply_to_tools=bool(value.get("apply_to_tools", False)),
            )
        return cls(policies)

    def get_policy(self, name: str) -> Policy:
        if name in self._policies:
            return self._policies[name]
        logger.warning("Unknown guardrail policy %r; falling back to 'default'", name)
        return self._policies["default"]

    def set_judge(self, judge: JudgeClient) -> None:
        """Give every LLM-judge rule the client it uses to call a model."""
        for policy in self._policies.values():
            for rule in (*policy.input_rules, *policy.output_rules):
                if isinstance(rule, LlmJudgeRule):
                    rule.judge = judge

    def list_policies(self) -> list[str]:
        return list(self._policies)

    async def evaluate(
        self,
        policy_name: str,
        text: str,
        phase: Phase,
        *,
        message_count: int | None = None,
    ) -> GuardrailResult:
        policy = self.get_policy(policy_name)
        rules = policy.rules_for(phase)
        matches = []
        for rule in rules:
            if isinstance(rule, LengthRule):
                match = rule.evaluate(text, message_count=message_count)
            elif isinstance(rule, LlmJudgeRule):
                match = await rule.aevaluate(text)
            else:
                match = rule.evaluate(text)
            if match is not None:
                matches.append(match)

        # Every rule, whatever its action, is evaluated against the ORIGINAL text, so the
        # offsets reported for BLOCK/FLAG/REDACT matches all refer to the same string.
        blocked = any(match.action == Action.BLOCK for match in matches)
        result_text = text
        if not blocked:
            matched_names = {match.rule_name for match in matches}
            result_text = _redact_once(text, [r for r in rules if r.name in matched_names])
        return GuardrailResult(
            policy=policy.name,
            phase=phase,
            matches=matches,
            text=result_text,
            blocked=blocked,
            flagged=any(match.action == Action.FLAG for match in matches),
            redacted=result_text != text,
        )

    def redact_text(self, policy_name: str, text: str, phase: Phase) -> str:
        return _redact_once(text, self.get_policy(policy_name).rules_for(phase))


def _redact_once(text: str, rules: list[Rule]) -> str:
    """Collect spans from every REDACT rule over the original text and substitute once.

    Chaining per-rule substitutions is unsafe: each pass would rescan text already rewritten
    by the previous one. Cross-rule overlaps use the same tie-break as ``resolve_overlaps``.
    """
    spans: list[Span] = []
    for rule in rules:
        if rule.action == Action.REDACT:
            spans.extend(rule.redaction_spans(text))
    return apply_spans(text, spans)


def _build_rules(entries: Any, policy: str, phase: str) -> list[Rule]:
    if not isinstance(entries, list):
        raise ConfigurationError(f"Rules for policy {policy!r} {phase} phase must be a list")
    rules: list[Rule] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigurationError(f"Rule {index} in policy {policy!r} must be a mapping")
        rule_type = entry.get("type")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            raise ConfigurationError(f"Rule {index} in policy {policy!r} requires a name")
        if not isinstance(rule_type, str) or rule_type not in _SUPPORTED_RULE_TYPES:
            raise ConfigurationError(f"Unknown guardrail rule type {rule_type!r} for rule {name!r}")
        try:
            action = Action(entry.get("action", "flag"))
            severity = Severity(entry.get("severity", "medium"))
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(
                f"Invalid action or severity for guardrail rule {name!r}: {exc}"
            ) from exc
        try:
            rule = _make_rule(rule_type, name, action, severity, entry)
        except (KeyError, TypeError, ValueError, re.error) as exc:
            raise ConfigurationError(
                f"Invalid configuration for guardrail rule {name!r}: {exc}"
            ) from exc
        rules.append(rule)
    return rules


def _make_rule(
    rule_type: Any,
    name: str,
    action: Action,
    severity: Severity,
    entry: dict[str, Any],
) -> Rule:
    """Build one rule; ``rule_type`` is already one of ``_SUPPORTED_RULE_TYPES``."""
    if rule_type == "regex":
        return RegexRule(
            name,
            str(entry["pattern"]),
            action,
            severity,
            ignorecase=bool(entry.get("ignorecase", False)),
            multiline=bool(entry.get("multiline", False)),
            replacement=str(entry.get("replacement", "[REDACTED]")),
        )
    if rule_type == "denylist":
        terms = entry["terms"]
        if not isinstance(terms, list) or not all(isinstance(term, str) for term in terms):
            raise ValueError("denylist terms must be a list of strings")
        return DenylistRule(
            name,
            terms,
            action,
            severity,
            phrases=bool(entry.get("phrases", False)),
            replacement=str(entry.get("replacement", "[REDACTED]")),
        )
    if rule_type == "length":
        return LengthRule(
            name,
            action,
            severity,
            max_chars=entry.get("max_chars"),
            max_messages=entry.get("max_messages"),
        )
    if rule_type == "topic":
        return TopicRule(
            name,
            entry["keywords"],
            action,
            severity,
            min_hits=int(entry.get("min_hits", 1)),
        )
    if rule_type == "llm_judge":
        return LlmJudgeRule(
            str(entry["model"]),
            str(entry["prompt"]),
            float(entry.get("threshold", 0.5)),
            name=name,
            action=action,
            severity=severity,
            on_error=str(entry.get("on_error", "allow")),
        )
    return PiiRule(
        name,
        entry.get("entities", ()),
        action,
        severity,
        mode=str(entry.get("mode", "full")),
    )
