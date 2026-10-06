"""Prometheus metrics for RAG retrieval and ingestion, and for MCP tool calls."""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.integration.conftest import metric_value


def test_rag_retrieval_and_ingestion_are_measured(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    before = {
        "ingest": metric_value(client, "aigw_rag_ingestions_total", outcome="ready", mode="inline"),
        "hit": metric_value(
            client, "aigw_rag_retrievals_total", search_mode="hybrid", outcome="hit"
        ),
        "empty": metric_value(
            client, "aigw_rag_retrievals_total", search_mode="vector", outcome="empty"
        ),
        "seconds": metric_value(client, "aigw_rag_retrieval_seconds_count", search_mode="hybrid"),
    }
    collection = client.post("/v1/rag/collections", json={"name": "m"}, headers=auth_headers).json()
    client.post(
        f"/v1/rag/collections/{collection['id']}/documents",
        json={"content": "Paris is in France.", "source": "p.md"},
        headers=auth_headers,
    )
    for mode, min_score in (("hybrid", 0.0), ("vector", 0.99)):
        client.post(
            "/v1/rag/search",
            json={
                "collection_id": collection["id"],
                "query": "Paris",
                "search_mode": mode,
                "min_score": min_score,
            },
            headers=auth_headers,
        )
    assert (
        metric_value(client, "aigw_rag_ingestions_total", outcome="ready", mode="inline")
        == before["ingest"] + 1
    )
    assert (
        metric_value(client, "aigw_rag_retrievals_total", search_mode="hybrid", outcome="hit")
        == before["hit"] + 1
    )
    assert (
        metric_value(client, "aigw_rag_retrievals_total", search_mode="vector", outcome="empty")
        == before["empty"] + 1
    )
    assert (
        metric_value(client, "aigw_rag_retrieval_seconds_count", search_mode="hybrid")
        == before["seconds"] + 1
    )


def test_mcp_tool_calls_and_breaker_state_are_measured(
    client: TestClient, auth_headers: dict[str, str], fake_mcp_url: str
) -> None:
    client.post(
        "/v1/mcp/servers",
        json={"name": "metered", "transport": "http", "url": fake_mcp_url},
        headers=auth_headers,
    )
    before = metric_value(
        client, "aigw_mcp_tool_calls_total", server="metered", tool="metered__add", status="ok"
    )
    client.post(
        "/v1/mcp/tools/call",
        json={"name": "metered__add", "arguments": {"a": 1, "b": 2}},
        headers=auth_headers,
    )
    assert (
        metric_value(
            client, "aigw_mcp_tool_calls_total", server="metered", tool="metered__add", status="ok"
        )
        == before + 1
    )
    assert metric_value(client, "aigw_mcp_tool_call_seconds_count", server="metered") >= 1
    assert metric_value(client, "aigw_mcp_breaker_open", server="metered") == 0
