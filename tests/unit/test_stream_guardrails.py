"""The stream redactor: redaction that holds across chunk boundaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.errors import GuardrailViolationError
from app.guardrails.registry import GuardrailRegistry
from app.guardrails.stream import StreamRedactor

SECRET = "sk-abcdefghijklmnopqrstuvwxyz123456"

POLICIES = """
policies:
  default:
    output:
      - name: redact-keys
        type: regex
        pattern: "sk-[A-Za-z0-9]{16,}"
        action: redact
        replacement: "[KEY]"
  strict:
    output:
      - name: block-forbidden
        type: denylist
        terms: [forbidden]
        action: block
"""


@pytest.fixture
def registry(tmp_path: Path) -> GuardrailRegistry:
    path = tmp_path / "guardrails.yaml"
    path.write_text(POLICIES)
    return GuardrailRegistry.load(str(path))


def _stream(redactor: StreamRedactor, text: str, size: int) -> tuple[str, list[str]]:
    releases = [redactor.feed(text[i : i + size]) for i in range(0, len(text), size)]
    releases.append(redactor.finish())
    return "".join(releases), releases


@pytest.mark.parametrize("size", [1, 3, 7, 50])
def test_secret_split_across_chunks_is_redacted(registry: GuardrailRegistry, size: int) -> None:
    redactor = StreamRedactor(registry, "default", holdback=64)
    text = f"Here is the key {SECRET} - keep it safe. " * 3
    output, _ = _stream(redactor, text, size)
    assert SECRET[:10] not in output
    assert output == text.replace(SECRET, "[KEY]")


def test_text_is_released_once_it_is_older_than_the_holdback(
    registry: GuardrailRegistry,
) -> None:
    redactor = StreamRedactor(registry, "default", holdback=10)
    assert redactor.feed("0123456789") == ""
    assert redactor.feed("abcde") == "01234"
    assert redactor.finish() == "56789abcde"


def test_a_match_straddling_the_release_point_is_held_whole(
    registry: GuardrailRegistry,
) -> None:
    redactor = StreamRedactor(registry, "default", holdback=5)
    released = redactor.feed("prefix " + SECRET[:20])
    released += redactor.feed(SECRET[20:] + " suffix and more text here")
    released += redactor.finish()
    assert released == "prefix [KEY] suffix and more text here"


def test_block_rule_stops_the_stream(registry: GuardrailRegistry) -> None:
    redactor = StreamRedactor(registry, "strict", holdback=0)
    assert redactor.feed("all good ") == "all good "
    with pytest.raises(GuardrailViolationError):
        redactor.feed("this is forbidden")


def test_policy_without_output_rules_is_inactive(tmp_path: Path) -> None:
    path = tmp_path / "guardrails.yaml"
    path.write_text("policies:\n  default:\n    input: []\n")
    redactor = StreamRedactor(GuardrailRegistry.load(str(path)), "default", holdback=64)
    assert redactor.active is False
