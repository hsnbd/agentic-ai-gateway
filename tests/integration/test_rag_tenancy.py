"""RAG collections are isolated per team (or per key, for keys without a team).

Global collections, created by operators, are readable by everyone but only
operators can change them. Another tenant's collection is reported as not
found, so its existence is not confirmed.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.integration.conftest import create_virtual_key


def _bearer(secret: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {secret}"}


def _team(client: TestClient, admin_headers: dict[str, str], name: str) -> str:
    response = client.post("/admin/api/teams", json={"name": name}, headers=admin_headers)
    assert response.status_code == 201, response.text
    team_id: str = response.json()["id"]
    return team_id


@pytest.fixture
def tenants(client: TestClient, admin_headers: dict[str, str]) -> dict[str, dict[str, str]]:
    """Two keys in team A, one in team B, and one key with no team."""
    team_a = _team(client, admin_headers, "tenant-a")
    team_b = _team(client, admin_headers, "tenant-b")
    keys = {
        "a1": create_virtual_key(client, admin_headers, name="a1", team_id=team_a)[0],
        "a2": create_virtual_key(client, admin_headers, name="a2", team_id=team_a)[0],
        "b": create_virtual_key(client, admin_headers, name="b", team_id=team_b)[0],
        "solo": create_virtual_key(client, admin_headers, name="solo")[0],
    }
    return {name: _bearer(secret) for name, secret in keys.items()}


def _create(client: TestClient, headers: dict[str, str], name: str) -> dict[str, Any]:
    response = client.post("/v1/rag/collections", json={"name": name}, headers=headers)
    assert response.status_code == 201, response.text
    body: dict[str, Any] = response.json()
    return body


def _ingest(client: TestClient, headers: dict[str, str], collection_id: str) -> Any:
    return client.post(
        f"/v1/rag/collections/{collection_id}/documents",
        json={"content": "Tenant secrets live here.", "source": "s.md"},
        headers=headers,
    )


def test_team_collections_are_shared_within_the_team_only(
    client: TestClient, tenants: dict[str, dict[str, str]]
) -> None:
    owned = _create(client, tenants["a1"], "team-a-docs")
    assert owned["owner_team_id"] and owned["owner_key_id"] is None
    assert _ingest(client, tenants["a1"], owned["id"]).status_code == 201

    # A teammate can read, search, and add to it.
    assert (
        client.get(f"/v1/rag/collections/{owned['id']}", headers=tenants["a2"]).status_code == 200
    )
    assert _ingest(client, tenants["a2"], owned["id"]).status_code in (200, 201)
    search = client.post(
        "/v1/rag/search",
        json={"collection_id": owned["id"], "query": "tenant secrets"},
        headers=tenants["a2"],
    )
    assert search.status_code == 200 and search.json()["results"]

    # Other tenants see nothing: not in the list, and every route says "not found".
    for outsider in ("b", "solo"):
        headers = tenants[outsider]
        listed = client.get("/v1/rag/collections", headers=headers).json()
        assert owned["id"] not in {item["id"] for item in listed}
        base = f"/v1/rag/collections/{owned['id']}"
        assert client.get(base, headers=headers).status_code == 404
        assert client.get(f"{base}/documents", headers=headers).status_code == 404
        assert client.get(f"{base}/chunks", headers=headers).status_code == 404
        assert _ingest(client, headers, owned["id"]).status_code == 404
        assert client.delete(base, headers=headers).status_code == 404
        searched = client.post(
            "/v1/rag/search",
            json={"collection_id": owned["id"], "query": "tenant secrets"},
            headers=headers,
        )
        assert searched.status_code == 404


def test_a_key_without_a_team_owns_its_collections(
    client: TestClient, tenants: dict[str, dict[str, str]]
) -> None:
    solo = _create(client, tenants["solo"], "solo-docs")
    assert solo["owner_key_id"] and solo["owner_team_id"] is None
    assert client.get(f"/v1/rag/collections/{solo['id']}", headers=tenants["a1"]).status_code == 404
    assert (
        client.delete(f"/v1/rag/collections/{solo['id']}", headers=tenants["solo"]).status_code
        == 200
    )


def test_global_collections_are_readable_by_all_but_changed_only_by_operators(
    client: TestClient,
    tenants: dict[str, dict[str, str]],
    auth_headers: dict[str, str],
    admin_headers: dict[str, str],
) -> None:
    shared = _create(client, auth_headers, "handbook")
    assert shared["owner_team_id"] is None and shared["owner_key_id"] is None
    assert _ingest(client, auth_headers, shared["id"]).status_code == 201

    for headers in tenants.values():
        assert client.get(f"/v1/rag/collections/{shared['id']}", headers=headers).status_code == 200
    refused = _ingest(client, tenants["a1"], shared["id"])
    assert refused.status_code == 403
    assert refused.json()["error"]["code"] == "permission_denied"
    assert (
        client.delete(f"/v1/rag/collections/{shared['id']}", headers=tenants["b"]).status_code
        == 403
    )

    # Console admins see every tenant's collections.
    private = _create(client, tenants["b"], "b-private")
    listed = {c["id"] for c in client.get("/v1/rag/collections", headers=admin_headers).json()}
    assert {shared["id"], private["id"]} <= listed


def test_names_are_unique_per_owner_not_globally(
    client: TestClient, tenants: dict[str, dict[str, str]]
) -> None:
    _create(client, tenants["a1"], "notes")
    _create(client, tenants["b"], "notes")  # another tenant may reuse the name
    clash = client.post("/v1/rag/collections", json={"name": "notes"}, headers=tenants["a2"])
    assert clash.status_code == 409


def test_chat_grounding_respects_ownership(
    client: TestClient, tenants: dict[str, dict[str, str]]
) -> None:
    owned = _create(client, tenants["a1"], "grounding")
    _ingest(client, tenants["a1"], owned["id"])
    body = {
        "model": "test-model",
        "messages": [{"role": "user", "content": "tenant secrets?"}],
        "aigw": {"rag": {"collection_id": owned["id"]}},
    }
    assert client.post("/v1/chat/completions", json=body, headers=tenants["a2"]).status_code == 200
    denied = client.post("/v1/chat/completions", json=body, headers=tenants["b"])
    assert denied.status_code == 404
