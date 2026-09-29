"""A minimal, deterministic MCP server for tests and local evaluation.

Speaks JSON-RPC 2.0 over either MCP transport the gateway supports:

    uv run python scripts/fake_mcp_server.py --port 4200   # streamable HTTP
    uv run python scripts/fake_mcp_server.py --stdio       # newline-delimited stdio

Tools:
  echo(text)      -> returns the text unchanged
  add(a, b)       -> returns a + b
  fail()          -> returns an MCP tool error (isError: true)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

PROTOCOL_VERSION = "2025-03-26"
SESSION_ID = "fake-mcp-session"

TOOLS: list[dict[str, Any]] = [
    {
        "name": "echo",
        "description": "Echo the given text back.",
        "inputSchema": {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        },
    },
    {
        "name": "add",
        "description": "Add two numbers.",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    {
        "name": "fail",
        "description": "Always reports a tool error.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


def _text(value: str, *, error: bool = False) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": value}], "isError": error}


def handle(message: dict[str, Any]) -> dict[str, Any] | None:
    """Answer one JSON-RPC message; notifications (no id) get no reply."""
    method = message.get("method")
    params = message.get("params") or {}
    if "id" not in message:
        return None

    result: dict[str, Any]
    if method == "initialize":
        result = {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": "fake-mcp", "version": "1.0.0"},
        }
    elif method == "tools/list":
        result = {"tools": TOOLS}
    elif method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        if name == "echo":
            result = _text(str(args.get("text", "")))
        elif name == "add":
            total = float(args.get("a", 0)) + float(args.get("b", 0))
            result = _text(str(int(total) if total.is_integer() else total))
        elif name == "fail":
            result = _text("tool failed on purpose", error=True)
        else:
            return {
                "jsonrpc": "2.0",
                "id": message["id"],
                "error": {"code": -32602, "message": f"Unknown tool: {name}"},
            }
    elif method == "ping":
        result = {}
    else:
        return {
            "jsonrpc": "2.0",
            "id": message["id"],
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }
    return {"jsonrpc": "2.0", "id": message["id"], "result": result}


def create_app() -> FastAPI:
    app = FastAPI(title="fake-mcp")

    @app.post("/")
    @app.post("/mcp")
    async def rpc(request: Request) -> Response:
        reply = handle(await request.json())
        if reply is None:
            return Response(status_code=202)
        return JSONResponse(reply, headers={"Mcp-Session-Id": SESSION_ID})

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    return app


def serve_stdio() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        reply = handle(json.loads(line))
        if reply is not None:
            sys.stdout.write(json.dumps(reply) + "\n")
            sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=4200)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--stdio", action="store_true", help="Serve over stdin/stdout")
    args = parser.parse_args()
    if args.stdio:
        serve_stdio()
        return
    import uvicorn

    uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
