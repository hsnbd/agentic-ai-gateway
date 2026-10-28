"""What an evaluation asks: the gateway's chat API, or an application over HTTP.

Each target turns one case into an ``Observation``: the answer, the context it
was grounded in, and the tools that ran, which is what DeepEval scores.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx


@dataclass
class Observation:
    output: str = ""
    retrieval_context: list[str] = field(default_factory=list)
    tools_called: list[str] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str | None = None


class Target(Protocol):
    def run(self, case: dict[str, Any]) -> Observation: ...


class GatewayChatTarget:
    """``POST /v1/chat/completions``, with the dataset's (and case's) ``aigw`` controls.

    ``aigw.rag`` grounds the answer and its sources become the retrieval
    context; ``aigw.mcp`` runs the agent loop and the executed tools are
    reported as ``tools_called``. A case's inline ``context`` (without RAG) is
    sent as a grounding system prompt, the way RAG would place it.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        system: str | None = None,
        aigw: dict[str, Any] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self.system = system
        self.aigw = aigw or {}
        self.client = client or httpx.Client(timeout=120)
        self.url = f"{base_url.rstrip('/')}/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def run(self, case: dict[str, Any]) -> Observation:
        system = [self.system] if self.system else []
        if case.get("context") and "rag" not in {**self.aigw, **case.get("aigw", {})}:
            system.append("Answer using only this context:\n\n" + "\n\n".join(case["context"]))
        messages = [{"role": "system", "content": "\n\n".join(system)}] if system else []
        messages.append({"role": "user", "content": case["input"]})
        body: dict[str, Any] = {
            "model": case.get("model", self.model),
            "messages": messages,
            # A cached answer would score the cache, not the model.
            "aigw": {"no_cache": True, **self.aigw, **case.get("aigw", {})},
        }
        if "user" in case:
            body["user"] = case["user"]
        started = time.perf_counter()
        response = self.client.post(self.url, json=body, headers=self.headers)
        elapsed = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            error = f"HTTP {response.status_code}: {response.text}"
            return Observation(latency_ms=elapsed, error=error)
        payload = response.json()
        extras = payload.get("aigw") or {}
        return Observation(
            output=payload["choices"][0]["message"].get("content") or "",
            retrieval_context=[source["text"] for source in extras.get("sources", [])],
            tools_called=[item["name"] for item in extras.get("tool_executions", [])],
            latency_ms=elapsed,
        )


class HttpTarget:
    """Any JSON HTTP application, described in the dataset.

    ``start`` optionally opens a conversation first (its ``id_path`` value is
    available as ``{id}``); ``request`` sends the case input. Strings in paths
    and bodies may use ``{input}``, ``{id}``, and any field of the case's
    ``vars``. ``output_path``, ``context_path`` and ``tools_path`` select from
    the reply, with ``a.b``, ``a[0]`` and ``a[*].b`` steps; a case may override
    any of them (e.g. to read the tools of an action awaiting approval).
    """

    def __init__(
        self,
        *,
        base_url: str,
        request: dict[str, Any],
        output_path: str,
        context_path: str | None = None,
        tools_path: str | None = None,
        start: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.request = request
        self.start = start
        self.output_path = output_path
        self.context_path = context_path
        self.tools_path = tools_path
        self.headers = headers or {}
        self.client = client or httpx.Client(timeout=120)

    def _send(self, spec: dict[str, Any], values: dict[str, Any]) -> Any:
        response = self.client.request(
            spec.get("method", "POST"),
            self.base_url + fill(spec["path"], values),
            json=fill(spec.get("body"), values),
            headers=self.headers,
        )
        response.raise_for_status()
        return response.json()

    def run(self, case: dict[str, Any]) -> Observation:
        values: dict[str, Any] = {**case.get("vars", {}), "input": case["input"]}
        started = time.perf_counter()
        try:
            if self.start is not None:
                start = {**self.start, "body": case.get("start_body", self.start.get("body"))}
                opened = self._send(start, values)
                values["id"] = select(opened, self.start.get("id_path", "id"))[0]
            reply = self._send(self.request, values)
        except (httpx.HTTPError, IndexError, KeyError, ValueError) as exc:
            elapsed = (time.perf_counter() - started) * 1000
            return Observation(latency_ms=elapsed, error=f"{type(exc).__name__}: {exc}")
        output = select(reply, case.get("output_path", self.output_path))
        return Observation(
            output=str(output[0]) if output else "",
            retrieval_context=[
                str(v) for v in select(reply, case.get("context_path", self.context_path))
            ],
            tools_called=[str(v) for v in select(reply, case.get("tools_path", self.tools_path))],
            latency_ms=(time.perf_counter() - started) * 1000,
        )


def build_target(spec: dict[str, Any], *, base_url: str, api_key: str, model: str) -> Target:
    """The target a dataset's ``target`` section describes."""
    options = {key: value for key, value in spec.items() if key != "type"}
    kind = spec.get("type", "gateway")
    if kind == "gateway":
        return GatewayChatTarget(
            base_url=options.pop("base_url", base_url),
            api_key=options.pop("api_key", api_key),
            model=options.pop("model", model),
            **options,
        )
    if kind == "http":
        return HttpTarget(**options)
    raise ValueError(f"Unknown target type {kind!r}; use 'gateway' or 'http'")


_STEP = re.compile(r"([^.\[\]]+)|\[(\*|\d+)\]")


def select(data: Any, path: str | None) -> list[Any]:
    """Every value at ``path``; missing steps select nothing rather than failing."""
    if not path:
        return []
    current = [data]
    for name, index in _STEP.findall(path):
        following: list[Any] = []
        for value in current:
            if name:
                if isinstance(value, dict) and name in value:
                    following.append(value[name])
            elif isinstance(value, list):
                if index == "*":
                    following.extend(value)
                elif int(index) < len(value):
                    following.append(value[int(index)])
        current = following
    return current


def fill(template: Any, values: dict[str, Any]) -> Any:
    """Substitute ``{name}`` placeholders throughout a JSON-like template.

    A string that is exactly one placeholder takes the value's own type.
    """
    if isinstance(template, str):
        whole = re.fullmatch(r"\{(\w+)\}", template)
        if whole and whole.group(1) in values:
            return values[whole.group(1)]
        return re.sub(
            r"\{(\w+)\}",
            lambda m: str(values[m.group(1)]) if m.group(1) in values else m.group(0),
            template,
        )
    if isinstance(template, dict):
        return {key: fill(value, values) for key, value in template.items()}
    if isinstance(template, list):
        return [fill(value, values) for value in template]
    return template
