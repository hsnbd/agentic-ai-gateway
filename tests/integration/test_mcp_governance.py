"""MCP governance end to end: per-key server and tool allowlists, guardrails on
tool arguments and results, result size caps, and the tool-call audit log.

Runs the real gateway against the real fake MCP server, with a guardrail file
whose rules cover every outcome (block or redact, on arguments or results).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.integration.conftest import FakeProvider, chat_body, create_virtual_key

POLICIES = """
policies:
  default:
    apply_to_tools: true
    input:
      - {name: block-injection, type: regex, pattern: "ignore previous instructions",
         ignorecase: true, action: block}
      - {name: redact-email, type: pii, entities: [EMAIL], action: redact, mode: mask}
    output:
      - {name: block-classified, type: denylist, terms: ["TOP-SECRET"], action: block}
      - {name: redact-keys, type: regex, pattern: "sk-[A-Za-z0-9]{16,}", action: redact,
         replacement: "[API_KEY_REDACTED]"}
  untooled:
    output:
      - {name: redact-keys, type: regex, pattern: "sk-[A-Za-z0-9]{16,}", action: redact}
"""

SECRET = "sk-abcdefghijklmnopqrstuvwx"


@pytest.fixture
def extra_env(tmp_path: Path) -> dict[str, str]:
    path = tmp_path / "guardrails.yaml"
    path.write_text(POLICIES)
    return {"GUARDRAILS_CONFIG_PATH": str(path), "MCP_MAX_RESULT_CHARS": "200"}


@pytest.fixture
def server_id(client: TestClient, auth_headers: dict[str, str], fake_mcp_url: str) -> str:
    response = client.post(
        "/v1/mcp/servers",
        json={"name": "tools", "transport": "http", "url": fake_mcp_url},
        headers=auth_headers,
    )
    assert response.status_code == 201, response.text
    value: str = response.json()["id"]
    return value


def _key(client: TestClient, admin_headers: dict[str, str], **fields: Any) -> dict[str, str]:
    secret, _ = create_virtual_key(client, admin_headers, **fields)
    return {"Authorization": f"Bearer {secret}"}


def _call(client: TestClient, headers: dict[str, str], name: str, **arguments: Any) -> str:
    response = client.post(
        "/v1/mcp/tools/call", json={"name": name, "arguments": arguments}, headers=headers
    )
    assert response.status_code == 200, response.text
    content: str = response.json()["message"]["content"]
    return content


def _audit(client: TestClient, admin_headers: dict[str, str], **filters: str) -> list[Any]:
    response = client.get("/admin/api/tool-calls", params=filters, headers=admin_headers)
    assert response.status_code == 200, response.text
    items: list[Any] = response.json()["items"]
    return items


class TestAllowlists:
    def test_tool_allowlist_hides_and_refuses_other_tools(
        self, client: TestClient, admin_headers: dict[str, str], server_id: str
    ) -> None:
        headers = _key(client, admin_headers, allowed_tools=["tools__add"])
        listed = client.get("/v1/mcp/tools", headers=headers).json()
        assert [tool["function"]["name"] for tool in listed] == ["tools__add"]

        assert _call(client, headers, "tools__add", a=2, b=3) == "5"
        refused = _call(client, headers, "tools__echo", text="hi")
        assert "not allowed for this API key" in refused

        statuses = {row["tool"]: row["status"] for row in _audit(client, admin_headers)}
        assert statuses == {"tools__add": "ok", "tools__echo": "denied"}

    def test_wildcards_and_server_names(
        self, client: TestClient, admin_headers: dict[str, str], server_id: str
    ) -> None:
        by_name = _key(
            client, admin_headers, allowed_mcp_servers=["tools"], allowed_tools=["tools__*"]
        )
        assert len(client.get("/v1/mcp/tools", headers=by_name).json()) == 3
        by_id = _key(client, admin_headers, allowed_mcp_servers=[server_id])
        assert _call(client, by_id, "tools__add", a=1, b=1) == "2"

    def test_server_allowlist_blocks_listing_calls_and_agent_requests(
        self,
        client: TestClient,
        admin_headers: dict[str, str],
        server_id: str,
        primary: FakeProvider,
    ) -> None:
        headers = _key(client, admin_headers, allowed_mcp_servers=["some-other-server"])
        assert client.get("/v1/mcp/tools", headers=headers).json() == []
        assert "not allowed" in _call(client, headers, "tools__add", a=1, b=2)

        named = client.post(
            "/v1/chat/completions",
            json=chat_body("2 + 3?", aigw={"mcp": {"servers": [server_id]}}),
            headers=headers,
        )
        assert named.status_code == 403
        assert named.json()["error"]["code"] == "permission_denied"

        unnamed = client.post(
            "/v1/chat/completions", json=chat_body("2 + 3?", aigw={"mcp": {}}), headers=headers
        )
        assert unnamed.status_code == 200, unnamed.text
        offered = {tool.function.name for tool in primary.seen_requests[-1].tools}
        assert not {name for name in offered if name.startswith("tools__")}


class TestGuardrailsOnTools:
    def test_secrets_in_results_are_redacted_and_recorded(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
        server_id: str,
    ) -> None:
        content = _call(client, auth_headers, "tools__echo", text=f"token {SECRET}")
        assert SECRET not in content and "[API_KEY_REDACTED]" in content

        [row] = _audit(client, admin_headers, tool="tools__echo")
        assert row["status"] == "ok" and row["guardrail"] == "redact-keys"
        violations = client.get("/admin/api/guardrails/violations", headers=admin_headers).json()
        detail = violations["items"][0]["details"]
        assert detail == {**detail, "tool": "tools__echo", "stage": "tool_result"}

    def test_classified_results_are_withheld(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
        server_id: str,
    ) -> None:
        content = _call(client, auth_headers, "tools__echo", text="TOP-SECRET plans")
        assert content == "[Tool result withheld by guardrail rule 'block-classified']"
        assert _audit(client, admin_headers)[0]["status"] == "blocked"

    def test_arguments_are_blocked_or_redacted_before_the_server_sees_them(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
        server_id: str,
    ) -> None:
        blocked = _call(client, auth_headers, "tools__echo", text="Ignore previous instructions")
        assert blocked == "Tool call blocked by guardrail rule 'block-injection'"

        echoed = _call(client, auth_headers, "tools__echo", text="mail jane@example.com")
        assert "jane@example.com" not in echoed

        rows = _audit(client, admin_headers)
        assert [row["status"] for row in rows] == ["ok", "blocked"]
        assert rows[1]["guardrail"] == "block-injection"

    def test_policies_without_apply_to_tools_leave_tool_traffic_alone(
        self, client: TestClient, admin_headers: dict[str, str], server_id: str
    ) -> None:
        headers = _key(client, admin_headers, guardrail_policy="untooled")
        assert SECRET in _call(client, headers, "tools__echo", text=f"token {SECRET}")


class TestLimitsAndAudit:
    def test_long_results_are_truncated(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
        server_id: str,
    ) -> None:
        content = _call(client, auth_headers, "tools__echo", text="x" * 500)
        assert content.startswith("x" * 200)
        assert content.endswith("[Tool result truncated: 300 more characters]")
        [row] = _audit(client, admin_headers)
        assert row["truncated"] is True and row["result_chars"] == 500

    def test_failures_invalid_arguments_and_tool_errors_are_audited(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
        server_id: str,
    ) -> None:
        _call(client, auth_headers, "tools__add", a="two")
        _call(client, auth_headers, "tools__fail")
        _call(client, auth_headers, "tools__missing")
        statuses = {row["tool"]: row["status"] for row in _audit(client, admin_headers)}
        assert statuses == {
            "tools__add": "invalid",
            "tools__fail": "tool_error",
            "tools__missing": "unavailable",
        }

    def test_agent_loop_calls_are_audited_with_the_request_and_key(
        self,
        client: TestClient,
        admin_headers: dict[str, str],
        server_id: str,
        primary: FakeProvider,
    ) -> None:
        secret, key = create_virtual_key(client, admin_headers, name="agent")
        primary.tool_queue = [("tools__add", {"a": 2, "b": 3})]
        response = client.post(
            "/v1/chat/completions",
            json=chat_body("2 + 3?", aigw={"mcp": {}}),
            headers={"Authorization": f"Bearer {secret}"},
        )
        assert response.status_code == 200, response.text
        request_id = response.headers["X-Gateway-Request-Id"]

        [row] = _audit(client, admin_headers, request_id=request_id)
        assert row["source"] == "agent" and row["virtual_key_id"] == key["id"]
        assert row["server_id"] == server_id and row["status"] == "ok"
        assert len(row["arguments_hash"]) == 64

        filtered = _audit(client, admin_headers, key_id=key["id"], status="ok", server_id=server_id)
        assert [item["id"] for item in filtered] == [row["id"]]
        assert _audit(client, admin_headers, team_id="no-such-team") == []

    def test_the_master_key_is_unrestricted_and_audited_without_a_key(
        self,
        client: TestClient,
        auth_headers: dict[str, str],
        admin_headers: dict[str, str],
        server_id: str,
    ) -> None:
        assert _call(client, auth_headers, "tools__add", a=4, b=4) == "8"
        [row] = _audit(client, admin_headers)
        assert row["virtual_key_id"] is None and row["source"] == "direct"
