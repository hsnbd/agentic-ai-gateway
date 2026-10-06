"""Tool-call governance helpers: scope matching, argument redaction, and audit failures."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from app.core.schemas import ToolCall, ToolDef
from app.tools.executor import ToolExecutor, _map_strings
from app.tools.governance import UNRESTRICTED_SCOPE, McpScope, ToolCallContext


def test_scope_from_keys_and_matching() -> None:
    assert McpScope.for_key(None) is UNRESTRICTED_SCOPE
    key = SimpleNamespace(allowed_mcp_servers=["files"], allowed_tools=["files__read*"])
    scope = McpScope.for_key(key)
    assert scope.permits_server("id-1", "files") and scope.permits_server("files", None)
    assert not scope.permits_server("id-2", "billing")
    assert scope.permits_tool("files__read_text") and not scope.permits_tool("files__delete")
    assert McpScope.for_key(SimpleNamespace()) == UNRESTRICTED_SCOPE


def test_map_strings_walks_nested_values() -> None:
    value = {"to": "a@b.c", "cc": ["x@y.z", 3], "meta": {"n": None, "note": "hi"}}
    assert _map_strings(value, str.upper) == {
        "to": "A@B.C",
        "cc": ["X@Y.Z", 3],
        "meta": {"n": None, "note": "HI"},
    }


class _Registry:
    async def tools_for(self, server_ids: Any = None) -> list[ToolDef]:
        return [ToolDef.model_validate({"type": "function", "function": {"name": "s__t"}})]

    def resolve(self, name: str) -> tuple[str, str]:
        return "s", "t"

    def server_name(self, server_id: str) -> str:
        return server_id

    async def call_tool(self, server_id: str, name: str, arguments: Any) -> dict[str, Any]:
        return {"content": [{"type": "text", "text": "done"}]}


class _BrokenDb:
    def session(self) -> Any:
        raise RuntimeError("database is down")


async def test_a_failed_audit_write_is_logged_not_raised(caplog: pytest.LogCaptureFixture) -> None:
    state = SimpleNamespace(db=_BrokenDb(), components={}, settings=None)
    executor = ToolExecutor(
        _Registry(),  # type: ignore[arg-type]
        call_context=ToolCallContext(state=state, source="direct"),
    )
    message = await executor.execute(ToolCall(name="s__t", arguments="{}"))
    assert message.content == "done"
    assert "Could not record MCP tool call s__t" in caplog.text
