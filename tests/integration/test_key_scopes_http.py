"""Key scopes over HTTP: collection allowlists, multi-collection search, and
tool arguments bound to the end user, against the real fake MCP server.

These are the building blocks an application uses to give each of its user
roles a key of its own (see docs/agentic-apps.md).
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.integration.conftest import create_virtual_key


def _bearer(client: TestClient, admin_headers: dict[str, str], **fields: Any) -> dict[str, str]:
    secret, _ = create_virtual_key(client, admin_headers, **fields)
    return {"Authorization": f"Bearer {secret}"}


def _collection(client: TestClient, headers: dict[str, str], name: str, text: str) -> str:
    created = client.post("/v1/rag/collections", json={"name": name}, headers=headers)
    assert created.status_code == 201, created.text
    collection_id: str = created.json()["id"]
    ingested = client.post(
        f"/v1/rag/collections/{collection_id}/documents",
        json={"content": text, "source": f"{name}.md"},
        headers=headers,
    )
    assert ingested.status_code == 201, ingested.text
    return collection_id


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


def test_collection_allowlist_and_multi_collection_search(
    client: TestClient, auth_headers: dict[str, str], admin_headers: dict[str, str]
) -> None:
    public = _collection(client, auth_headers, "public-faq", "Admission opens in January.")
    staff = _collection(client, auth_headers, "staff-notes", "Grading rubric for staff.")
    scoped = _bearer(client, admin_headers, allowed_collections=["public-*"])

    listed = client.get("/v1/rag/collections", headers=scoped).json()
    assert [c["name"] for c in listed] == ["public-faq"]
    hidden = client.post(
        "/v1/rag/search", json={"collection_id": staff, "query": "grading"}, headers=scoped
    )
    assert hidden.status_code == 404

    both = client.post(
        "/v1/rag/search",
        json={"collection_ids": [public, staff], "query": "admission", "top_k": 5},
        headers=auth_headers,
    )
    assert both.status_code == 200, both.text
    body = both.json()
    assert body["collection_id"] == public and body["collection_ids"] == [public, staff]
    assert {r["metadata"]["collection_id"] for r in body["results"]} == {public, staff}

    # One forbidden collection makes the whole search "not found".
    mixed = client.post(
        "/v1/rag/search",
        json={"collection_ids": [public, staff], "query": "admission"},
        headers=scoped,
    )
    assert mixed.status_code == 404
    assert client.post("/v1/rag/search", json={"query": "x"}, headers=scoped).status_code == 422


def test_bound_arguments_are_hidden_and_filled_from_the_end_user(
    client: TestClient, admin_headers: dict[str, str], server_id: str
) -> None:
    headers = _bearer(
        client, admin_headers, tool_arg_bindings={"tools__echo": {"text": "end_user"}}
    )
    listed = {t["function"]["name"]: t for t in client.get("/v1/mcp/tools", headers=headers).json()}
    assert "text" not in listed["tools__echo"]["function"]["parameters"]["properties"]
    assert listed["tools__add"]["function"]["parameters"]["properties"].keys() == {"a", "b"}

    called = client.post(
        "/v1/mcp/tools/call",
        json={"name": "tools__echo", "arguments": {"text": "someone-else"}, "end_user": "u-42"},
        headers=headers,
    )
    assert called.status_code == 200, called.text
    assert "u-42" in called.json()["message"]["content"]
    assert "someone-else" not in called.json()["message"]["content"]

    refused = client.post(
        "/v1/mcp/tools/call", json={"name": "tools__echo", "arguments": {}}, headers=headers
    )
    assert "needs the caller's end_user" in refused.json()["message"]["content"]

    audit = client.get("/admin/api/tool-calls", headers=admin_headers).json()["items"]
    by_status = {row["status"]: row for row in audit}
    assert by_status["ok"]["bound_args"] == ["text"]
    assert by_status["unbound"]["bound_args"] == []


def test_key_admin_round_trips_the_new_scopes(
    client: TestClient, admin_headers: dict[str, str]
) -> None:
    created = client.post(
        "/admin/api/keys",
        json={
            "name": "role-key",
            "allowed_collections": ["public-*"],
            "tool_arg_bindings": {"crm__*": {"customer_id": "end_user", "secret": "none"}},
        },
        headers=admin_headers,
    )
    assert created.status_code == 201, created.text
    key = created.json()
    assert key["allowed_collections"] == ["public-*"]
    assert key["tool_arg_bindings"] == {"crm__*": {"customer_id": "end_user", "secret": "none"}}
    bad = client.patch(
        f"/admin/api/keys/{key['id']}",
        json={"tool_arg_bindings": {"crm__*": {"customer_id": "anyone"}}},
        headers=admin_headers,
    )
    assert bad.status_code == 422
    updated = client.patch(
        f"/admin/api/keys/{key['id']}",
        json={"allowed_collections": [], "tool_arg_bindings": {}},
        headers=admin_headers,
    )
    assert updated.json()["allowed_collections"] == [] and updated.json()["tool_arg_bindings"] == {}
