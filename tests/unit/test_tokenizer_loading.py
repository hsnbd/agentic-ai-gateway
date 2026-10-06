"""Tokenizers load in the background with a bounded wait, never hanging a request."""

from __future__ import annotations

import threading
from typing import Any

import pytest

from app.accounting import tokens


@pytest.fixture(autouse=True)
def _fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tokens, "_encodings", {})
    monkeypatch.setattr(tokens, "_loaders", {})
    monkeypatch.setattr(tokens, "_failed_at", {})


class _Encoder:
    def encode(self, text: str, disallowed_special: Any = ()) -> list[int]:
        return [0] * len(text.split())


def test_a_loaded_encoding_counts_exactly(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tokens.tiktoken, "get_encoding", lambda name: _Encoder())
    assert tokens.count_tokens("one two three", "gpt-4o") == 3
    assert tokens.count_tokens("x" * 40, "claude-sonnet-4-5") == 10  # unknown model: estimate


def test_a_slow_load_falls_back_to_an_estimate_without_waiting_again(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release = threading.Event()

    def slow(name: str) -> Any:
        release.wait(5)
        return _Encoder()

    monkeypatch.setattr(tokens.tiktoken, "get_encoding", slow)
    monkeypatch.setattr(tokens, "ENCODER_WAIT_SECONDS", 0.05)
    text = "alpha beta gamma delta epsilon"
    assert tokens.count_tokens(text, "gpt-4o") == len(text) // 4
    assert tokens.count_tokens(text, "gpt-4o") == len(text) // 4  # still loading: no second wait
    release.set()
    tokens._loaders["o200k_base"].join(2)
    assert tokens.count_tokens(text, "gpt-4o") == 5


def test_a_failed_load_is_retried_only_after_the_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    attempts: list[str] = []

    def offline(name: str) -> Any:
        attempts.append(name)
        raise OSError("network unreachable")

    monkeypatch.setattr(tokens.tiktoken, "get_encoding", offline)
    assert tokens.count_tokens("a b c d", "gpt-4o") == 1
    assert tokens.count_tokens("a b c d", "gpt-4o") == 1
    assert attempts == ["o200k_base"]

    monkeypatch.setattr(tokens, "ENCODER_RETRY_SECONDS", 0.0)
    monkeypatch.setattr(tokens.tiktoken, "get_encoding", lambda name: _Encoder())
    assert tokens.count_tokens("a b c d", "gpt-4o") == 4
