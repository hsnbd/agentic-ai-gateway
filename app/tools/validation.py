from __future__ import annotations

import ast
import json
import re
from typing import Any

from app.core.schemas import ToolDef


def validate_arguments(tool: ToolDef, raw_arguments: str) -> dict[str, Any]:
    schema = tool.function.parameters
    try:
        arguments = _parse_arguments(raw_arguments)
    except (ValueError, SyntaxError, json.JSONDecodeError) as exc:
        return {"error": f"Arguments are not valid JSON: {exc}", "expected_schema": schema}
    if not isinstance(arguments, dict):
        return {"error": "Tool arguments must be a JSON object", "expected_schema": schema}
    errors = _schema_errors(arguments, schema, "arguments")
    if errors:
        return {"error": "; ".join(errors), "expected_schema": schema}
    return arguments


def _parse_arguments(raw_arguments: str) -> Any:
    source = raw_arguments.strip()
    if source.startswith("```"):
        source = re.sub(r"\A```(?:json)?\s*|\s*```\Z", "", source, flags=re.IGNORECASE).strip()
    try:
        return json.loads(source)
    except json.JSONDecodeError:
        repaired = re.sub(r",\s*([}\]])", r"\1", source)
        repaired = re.sub(r"([{,]\s*)([A-Za-z_][\w-]*)(\s*:)", r'\1"\2"\3', repaired)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            try:
                return ast.literal_eval(repaired)
            except (ValueError, SyntaxError):
                repaired = re.sub(
                    r"(:\s*)([A-Za-z_][A-Za-z0-9_. -]*)(\s*[,}])",
                    lambda match: (
                        match.group(0)
                        if match.group(2) in {"true", "false", "null"}
                        else f'{match.group(1)}"{match.group(2).strip()}"{match.group(3)}'
                    ),
                    repaired,
                )
                return json.loads(repaired)


def _schema_errors(value: dict[str, Any], schema: dict[str, Any], path: str) -> list[str]:
    errors: list[str] = []
    required = schema.get("required", [])
    if isinstance(required, list):
        for key in required:
            if isinstance(key, str) and key not in value:
                errors.append(f"{path}.{key} is required")
    properties = schema.get("properties", {})
    if not isinstance(properties, dict):
        properties = {}
    for key, child in properties.items():
        if key not in value or not isinstance(child, dict):
            continue
        item = value[key]
        expected_type = child.get("type")
        if expected_type and not _matches_type(item, expected_type):
            errors.append(f"{path}.{key} must be {expected_type}")
            continue
        enum = child.get("enum")
        if isinstance(enum, list) and item not in enum:
            errors.append(f"{path}.{key} must be one of {enum!r}")
        if isinstance(item, dict):
            errors.extend(_schema_errors(item, child, f"{path}.{key}"))
    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        errors.append(f"{path} must be one of {enum!r}")
    return errors


def _matches_type(value: Any, expected_type: Any) -> bool:
    if isinstance(expected_type, list):
        return any(_matches_type(value, item) for item in expected_type)
    checks: dict[str, type | tuple[type, ...]] = {
        "object": dict,
        "array": list,
        "string": str,
        "integer": int,
        "number": (int, float),
        "boolean": bool,
        "null": type(None),
    }
    expected = checks.get(expected_type) if isinstance(expected_type, str) else None
    if expected is None:
        return True
    if expected_type in {"integer", "number"} and isinstance(value, bool):
        return False
    return isinstance(value, expected)
