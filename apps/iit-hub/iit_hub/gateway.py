"""The gateway's data-plane API, as the hub uses it, with one client per role key.

Chat goes through LangChain's ``ChatOpenAI`` (see ``llm.py``); this client
covers what ``ChatOpenAI`` does not: RAG search and governed MCP tool calls.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

#: How long a role's tool and collection lists are reused before asking again.
_TTL_SECONDS = 60.0


class GatewayError(RuntimeError):
    pass


class GatewayClient:
    def __init__(self, base_url: str, api_key: str, *, client: httpx.Client | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.http = client or httpx.Client(timeout=60)
        self._cache: dict[str, tuple[float, Any]] = {}

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self.http.request(
            method, f"{self.base_url}{path}", headers=self.headers, **kwargs
        )
        if response.status_code >= 400:
            raise GatewayError(f"{method} {path} -> {response.status_code}: {response.text}")
        return response.json()

    def _cached(self, name: str, fetch: Any) -> Any:
        hit = self._cache.get(name)
        if hit is not None and time.monotonic() - hit[0] < _TTL_SECONDS:
            return hit[1]
        value = fetch()
        self._cache[name] = (time.monotonic(), value)
        return value

    def tools(self) -> list[dict[str, Any]]:
        """MCP tools this key may call, with identity-bound arguments already hidden."""
        return self._cached("tools", lambda: self._request("GET", "/v1/mcp/tools"))  # type: ignore[no-any-return]

    def collection_ids(self) -> list[str]:
        """RAG collections this key may read (its allowlist decides)."""

        def fetch() -> list[str]:
            return [c["id"] for c in self._request("GET", "/v1/rag/collections")]

        return self._cached("collections", fetch)  # type: ignore[no-any-return]

    def search(self, query: str, *, top_k: int = 4) -> list[dict[str, Any]]:
        ids = self.collection_ids()
        if not ids:
            return []
        body = {"collection_ids": ids, "query": query, "top_k": top_k, "search_mode": "hybrid"}
        return self._request("POST", "/v1/rag/search", json=body)["results"]  # type: ignore[no-any-return]

    def call_tool(self, name: str, arguments: dict[str, Any], *, end_user: str | None) -> str:
        body: dict[str, Any] = {"name": name, "arguments": arguments}
        if end_user:
            body["end_user"] = end_user
        message = self._request("POST", "/v1/mcp/tools/call", json=body)["message"]
        return str(message.get("content") or "")
