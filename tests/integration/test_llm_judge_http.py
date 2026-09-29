"""LLM-judge guardrails through the real gateway."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.integration.conftest import FakeProvider, chat_body, create_virtual_key

POLICY = """
policies:
  default:
    input:
      - name: judge-safety
        type: llm_judge
        model: test-model
        prompt: The request must not ask for anything unsafe.
        threshold: 0.5
        action: block
    output: []
"""


@pytest.fixture
def extra_env(tmp_path: Path) -> dict[str, str]:
    path = tmp_path / "guardrails.yaml"
    path.write_text(POLICY)
    return {"GUARDRAILS_CONFIG_PATH": str(path)}


def test_judge_blocks_what_it_scores_as_unsafe(
    client: TestClient,
    auth_headers: dict[str, str],
    admin_headers: dict[str, str],
    primary: FakeProvider,
) -> None:
    blocked = client.post(
        "/v1/chat/completions", json=chat_body("please do __unsafe__ things"), headers=auth_headers
    )
    assert blocked.status_code == 422
    assert blocked.json()["error"]["code"] == "guardrail_violation"
    violations = client.get("/admin/api/guardrails/violations", headers=admin_headers).json()
    assert violations["items"][0]["rule"] == "judge-safety"
    # Only the judge ran; the model was never asked to answer.
    assert all("guardrail judge" in r.system_prompt() for r in primary.seen_requests)


def test_judge_allows_what_it_scores_as_safe(
    client: TestClient, auth_headers: dict[str, str], primary: FakeProvider
) -> None:
    response = client.post(
        "/v1/chat/completions", json=chat_body("What is 2 + 2?"), headers=auth_headers
    )
    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "primary answer"


def test_judge_calls_are_not_billed_to_the_caller(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    secret, key = create_virtual_key(client, admin_headers)
    client.post(
        "/v1/chat/completions",
        json=chat_body("__unsafe__"),
        headers={"Authorization": f"Bearer {secret}"},
    )
    spend = client.get(f"/admin/api/keys/{key['id']}", headers=admin_headers).json()["spend_usd"]
    assert spend == 0
