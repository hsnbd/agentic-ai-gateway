from __future__ import annotations

from typing import Any

from app.core.schemas import FunctionDef, Message, Role, ToolDef


def mcp_tool_to_tooldef(server_alias: str, tool: dict[str, Any]) -> ToolDef:
    name = tool.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError("MCP tool is missing a valid name")
    schema = tool.get("inputSchema")
    parameters = sanitize_schema(schema if isinstance(schema, dict) else {})
    if not parameters:
        parameters = {"type": "object", "properties": {}}
    description = tool.get("description", "")
    return ToolDef(
        function=FunctionDef(
            name=f"{server_alias}__{name}",
            description=description if isinstance(description, str) else "",
            parameters=parameters,
        )
    )


def tool_result_to_message(tool_call_id: str, result: dict[str, Any]) -> Message:
    blocks = result.get("content", [])
    output: list[str] = []
    if isinstance(blocks, list):
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text")
                if isinstance(text, str):
                    output.append(text)
            elif block_type == "image":
                mime = block.get("mimeType", "unknown MIME type")
                data = block.get("data")
                length = len(data) if isinstance(data, str) else 0
                output.append(f"[Image content ({mime}, {length} encoded characters) omitted]")
            elif block_type == "resource":
                resource = block.get("resource", {})
                if isinstance(resource, dict):
                    text = resource.get("text")
                    if isinstance(text, str):
                        output.append(text)
                    else:
                        uri = resource.get("uri", "unknown URI")
                        mime = resource.get("mimeType", "unknown MIME type")
                        blob = resource.get("blob")
                        length = len(blob) if isinstance(blob, str) else 0
                        output.append(
                            f"[Resource {uri} ({mime}; {length} encoded characters; non-text data)]"
                        )
    if result.get("isError") is True:
        output.insert(0, "Tool error reported by MCP server.")
    return Message(role=Role.TOOL, content="\n".join(output), tool_call_id=tool_call_id)


def sanitize_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove schema features unsupported by strict structured-output providers."""
    ignored = {"$schema", "$ref", "definitions", "$defs", "additionalProperties"}

    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items() if key not in ignored}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    sanitized = clean(schema)
    return sanitized if isinstance(sanitized, dict) else {}
