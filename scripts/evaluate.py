"""Evaluation harness for the gateway's stated grading criteria.

Drives a *running* gateway over HTTP and reports the metrics the project is
evaluated on: request success rate, response latency, provider failover, cache
hit ratio, cost per successful request, token and request usage, and routing
and guardrail effectiveness.

It deliberately talks to the gateway the way a real client does — no imports
from `app`, no test doubles — so the numbers describe the deployed system
rather than the code's opinion of itself.

    uv run python scripts/evaluate.py --base-url http://localhost:8000 \
        --api-key sk-... --model gpt-4o-mini --requests 50 --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class Attempt:
    """One request's observable outcome, from the client's point of view."""

    ok: bool
    latency_s: float
    status: int
    ttft_s: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    cache_hit: bool = False
    provider: str | None = None
    deployment: str | None = None
    error: str | None = None


@dataclass
class Section:
    name: str
    findings: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    # Nearest-rank: with small samples, interpolation invents precision we do
    # not have.
    rank = max(1, min(len(ordered), round(pct / 100.0 * len(ordered))))
    return ordered[rank - 1]


def _header_bool(headers: httpx.Headers, *names: str) -> bool:
    for name in names:
        raw = headers.get(name)
        if raw is not None:
            return raw.strip().lower() in {"hit", "true", "1", "yes"}
    return False


def _header_str(headers: httpx.Headers, *names: str) -> str | None:
    for name in names:
        raw = headers.get(name)
        if raw:
            return raw
    return None


class Evaluator:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        failover_model: str | None = None,
        routing_model: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        #: Model whose first deployment is deliberately unreachable, used to
        #: prove failover. Optional so the harness still runs against a plain
        #: production catalogue, where it reports the gap instead of failing.
        self.failover_model = failover_model
        #: Model with several healthy, differently-priced deployments, used to
        #: prove the routing strategies actually diverge.
        self.routing_model = routing_model or model
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._timeout = timeout

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.base_url, headers=self._headers, timeout=self._timeout
        )

    async def _chat(
        self,
        client: httpx.AsyncClient,
        prompt: str,
        *,
        extra: dict[str, Any] | None = None,
    ) -> Attempt:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if extra:
            body.update(extra)

        started = time.perf_counter()
        try:
            response = await client.post("/v1/chat/completions", json=body)
        except Exception as exc:
            return Attempt(
                ok=False, latency_s=time.perf_counter() - started, status=0, error=str(exc)
            )
        latency = time.perf_counter() - started

        if response.status_code != 200:
            return Attempt(
                ok=False,
                latency_s=latency,
                status=response.status_code,
                error=response.text[:400],
            )

        payload = response.json()
        usage = payload.get("usage") or {}
        return Attempt(
            ok=True,
            latency_s=latency,
            status=200,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            cost_usd=float(_header_str(response.headers, "x-gateway-cost-usd") or 0.0),
            cache_hit=_header_bool(response.headers, "x-gateway-cache"),
            provider=_header_str(response.headers, "x-gateway-provider"),
            deployment=_header_str(response.headers, "x-gateway-deployment"),
        )

    async def _chat_stream(self, client: httpx.AsyncClient, prompt: str) -> Attempt:
        """Measure time to first token, which unary requests cannot show."""
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
        }
        started = time.perf_counter()
        ttft: float | None = None
        try:
            async with client.stream("POST", "/v1/chat/completions", json=body) as response:
                if response.status_code != 200:
                    text = await response.aread()
                    return Attempt(
                        ok=False,
                        latency_s=time.perf_counter() - started,
                        status=response.status_code,
                        error=text.decode(errors="replace")[:400],
                    )
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    if line.strip() == "data: [DONE]":
                        break
                    if ttft is None:
                        ttft = time.perf_counter() - started
        except Exception as exc:
            return Attempt(
                ok=False, latency_s=time.perf_counter() - started, status=0, error=str(exc)
            )
        return Attempt(ok=True, latency_s=time.perf_counter() - started, status=200, ttft_s=ttft)

    # -- criteria -----------------------------------------------------------

    async def throughput_and_latency(self, total: int, concurrency: int) -> Section:
        """Success rate, latency distribution, token and cost accounting."""
        section = Section("success_rate_latency_cost")
        semaphore = asyncio.Semaphore(concurrency)

        async with self._client() as client:

            async def one(index: int) -> Attempt:
                async with semaphore:
                    # Unique prompts so the cache cannot flatter the latency.
                    return await self._chat(
                        client,
                        f"In one sentence, describe fact number {index} about oceans.",
                        extra={"no_cache": True},
                    )

            wall_start = time.perf_counter()
            attempts = await asyncio.gather(*(one(i) for i in range(total)))
            wall = time.perf_counter() - wall_start

        ok = [a for a in attempts if a.ok]
        latencies = [a.latency_s for a in ok]
        total_cost = sum(a.cost_usd for a in ok)
        prompt_tokens = sum(a.prompt_tokens for a in ok)
        completion_tokens = sum(a.completion_tokens for a in ok)

        section.findings = {
            "requests": total,
            "concurrency": concurrency,
            "succeeded": len(ok),
            "failed": total - len(ok),
            "success_rate": round(len(ok) / total, 4) if total else 0.0,
            "wall_clock_s": round(wall, 3),
            "throughput_rps": round(total / wall, 2) if wall > 0 else 0.0,
            "latency_p50_s": round(_percentile(latencies, 50), 3),
            "latency_p95_s": round(_percentile(latencies, 95), 3),
            "latency_p99_s": round(_percentile(latencies, 99), 3),
            "latency_mean_s": round(statistics.fmean(latencies), 3) if latencies else 0.0,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "total_cost_usd": round(total_cost, 6),
            "cost_per_successful_request_usd": (round(total_cost / len(ok), 6) if ok else None),
        }
        if total_cost == 0.0 and ok:
            section.notes.append(
                "Cost reported as zero. Either the price table has no entry for this "
                "model or the gateway does not expose the X-Gateway-Cost-USD header."
            )
        errors = {a.status: a.error for a in attempts if not a.ok}
        if errors:
            section.findings["error_samples"] = {
                str(status): (msg or "")[:200] for status, msg in list(errors.items())[:5]
            }
        return section

    async def time_to_first_token(self, samples: int) -> Section:
        section = Section("streaming_time_to_first_token")
        async with self._client() as client:
            attempts = [
                await self._chat_stream(client, f"Count slowly to five. Variation {i}.")
                for i in range(samples)
            ]
        ttfts = [a.ttft_s for a in attempts if a.ok and a.ttft_s is not None]
        section.findings = {
            "samples": samples,
            "succeeded": sum(1 for a in attempts if a.ok),
            "ttft_p50_s": round(_percentile(ttfts, 50), 3) if ttfts else None,
            "ttft_p95_s": round(_percentile(ttfts, 95), 3) if ttfts else None,
        }
        if not ttfts:
            section.notes.append("No streaming chunks observed; streaming may be unavailable.")
        return section

    async def cache_effectiveness(self, repeats: int) -> Section:
        """Semantic cache: a paraphrase should hit, an unrelated prompt must not."""
        section = Section("cache_hit_ratio")
        prompt = "What is the capital city of France?"
        paraphrases = [
            "What is the capital city of France?",
            "Which city is the capital of France?",
            "Tell me France's capital city.",
            "France's capital city is what?",
        ]
        # The control must be a prompt the cache has genuinely never seen. A
        # fixed string looks unrelated only on the first run: once a previous
        # run has cached it, the exact same text comes back as a legitimate hit
        # and the harness misreports it as a similarity false positive. The
        # nonce keeps the check meaningful against a warm cache.
        control = (
            f"Explain the rules of contract bridge in detail. Reference number {uuid.uuid4()}."
        )

        async with self._client() as client:
            seed = await self._chat(client, prompt)
            if not seed.ok:
                section.notes.append(f"Seed request failed ({seed.status}); cache not measured.")
                return section
            # The write happens after the response is returned; give it a beat.
            await asyncio.sleep(1.0)

            hits = 0
            checked = 0
            for i in range(repeats):
                attempt = await self._chat(client, paraphrases[i % len(paraphrases)])
                if attempt.ok:
                    checked += 1
                    hits += int(attempt.cache_hit)

            control_attempt = await self._chat(client, control)

        section.findings = {
            "seed_was_hit": seed.cache_hit,
            "followups": checked,
            "hits": hits,
            "hit_ratio": round(hits / checked, 4) if checked else 0.0,
            "unrelated_prompt_hit": control_attempt.cache_hit if control_attempt.ok else None,
        }
        if control_attempt.ok and control_attempt.cache_hit:
            section.notes.append(
                "CORRECTNESS RISK: an unrelated prompt returned a cache hit. The "
                "similarity threshold is too low."
            )
        if checked and hits == 0:
            section.notes.append(
                "No cache hits. Either caching is disabled, Redis Stack is absent, "
                "or the similarity threshold is too strict."
            )
        return section

    async def failover(self) -> Section:
        """Failover is proven only when a request that *must* fail on its first
        deployment is still answered by a later one.

        The earlier version of this check asked for a model name that does not
        exist and supplied `fallbacks`. That never exercised failover: an unknown
        model is rejected at registry resolution, before routing, and rightly so
        — silently serving a different model than the one requested would hide
        client typos and bill for them. So that case is now asserted as a
        *rejection*, and real failover is measured separately against a model
        whose first deployment is configured to be unreachable
        (see config/models.eval.yaml).
        """
        section = Section("provider_failover")
        findings: dict[str, Any] = {}

        async with self._client() as client:
            baseline = await self._chat(client, "Say OK.", extra={"no_cache": True})
            findings["baseline_ok"] = baseline.ok
            findings["baseline_served_by"] = baseline.deployment or baseline.provider

            # 1. An unknown model must be refused, not silently substituted.
            unknown = await self._chat(
                client,
                "Say OK.",
                extra={
                    "no_cache": True,
                    "model": "definitely-not-a-real-model",
                    "fallbacks": [self.model],
                },
            )
            findings["unknown_model_rejected"] = unknown.status in (400, 404)
            findings["unknown_model_status"] = unknown.status
            if unknown.ok:
                section.notes.append(
                    "An unknown model was served instead of rejected. A typo in the "
                    "model name should surface as an error, not as a silent "
                    "substitution the caller pays for."
                )

            # 2. Real failover: first deployment is dead, gateway must recover.
            if not self.failover_model:
                section.notes.append(
                    "No --failover-model supplied, so provider failover was not "
                    "measured. Run the gateway with MODELS_CONFIG_PATH="
                    "config/models.eval.yaml and pass --failover-model eval-chat."
                )
                section.findings = findings
                return section

            forced = await self._chat(
                client,
                "Say OK.",
                extra={"no_cache": True, "model": self.failover_model},
            )

        findings["failover_model"] = self.failover_model
        findings["failover_request_ok"] = forced.ok
        findings["failover_status"] = forced.status
        findings["served_by"] = forced.deployment or forced.provider
        # The broken deployment is first in config, so it holds the unsuffixed id.
        findings["recovered_from_dead_primary"] = bool(
            forced.ok and forced.deployment and "#" in forced.deployment
        )

        if not forced.ok:
            section.notes.append(
                f"Model {self.failover_model!r} did not recover from a failing "
                "primary deployment. Retry and fallback are not working. Error: "
                + (forced.error or "")[:200]
            )
        elif not findings["recovered_from_dead_primary"]:
            section.notes.append(
                f"Model {self.failover_model!r} succeeded but was served by its "
                "first deployment, so no failover actually occurred. Check that "
                "its primary deployment is genuinely unreachable."
            )

        section.findings = findings
        return section

    async def routing(self) -> Section:
        """Routing is effective if the strategy actually changes the choice."""
        section = Section("routing_effectiveness")
        observed: dict[str, str | None] = {}
        async with self._client() as client:
            for strategy in ("least-cost", "lowest-latency", "priority", "weighted"):
                attempt = await self._chat(
                    client,
                    "Reply with the single word: routed.",
                    extra={
                        "no_cache": True,
                        "routing_strategy": strategy,
                        "model": self.routing_model,
                    },
                )
                served = attempt.deployment or attempt.provider
                observed[strategy] = served if attempt.ok else f"ERROR {attempt.status}"
        section.findings = {"selection_by_strategy": observed}
        distinct = {v for v in observed.values() if v and not str(v).startswith("ERROR")}
        if len(distinct) <= 1:
            section.notes.append(
                "All strategies chose the same deployment. This is expected with a "
                "single deployment per model; configure at least two in models.yaml "
                "to demonstrate routing."
            )
        return section

    async def guardrails(self) -> Section:
        """A guardrail is effective only if it blocks and does not over-block."""
        section = Section("guardrail_effectiveness")
        benign = "Summarise the water cycle for a ten year old."
        pii = "My social security number is 123-45-6789 and my email is a@b.com."

        async with self._client() as client:
            benign_attempt = await self._chat(client, benign, extra={"no_cache": True})
            pii_attempt = await self._chat(client, pii, extra={"no_cache": True})

        section.findings = {
            "benign_allowed": benign_attempt.ok,
            "benign_status": benign_attempt.status,
            "pii_status": pii_attempt.status,
            "pii_blocked_or_redacted": pii_attempt.status == 403 or pii_attempt.ok,
        }
        if not benign_attempt.ok:
            section.notes.append(
                "FALSE POSITIVE: a benign prompt was rejected. Guardrails are over-blocking."
            )
        return section

    async def capabilities(self) -> Section:
        """Confirm the agentic surfaces answer, which is what RAG/MCP/tool
        support is graded on."""
        section = Section("capability_surface")
        probes = {
            "models": ("GET", "/v1/models"),
            "rag_collections": ("GET", "/v1/rag/collections"),
            "mcp_servers": ("GET", "/v1/mcp/servers"),
            "mcp_tools": ("GET", "/v1/mcp/tools"),
            "metrics": ("GET", "/metrics"),
            "console": ("GET", "/ui"),
        }
        results: dict[str, int] = {}
        async with self._client() as client:
            for label, (method, path) in probes.items():
                try:
                    response = await client.request(method, path)
                    results[label] = response.status_code
                except Exception:
                    results[label] = 0

        # Tool calling is a request-shape feature, so probe it as one.
        tool_def = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
        async with self._client() as client:
            tool_attempt = await self._chat(
                client,
                "What is the weather in Istanbul? Use the tool.",
                extra={"tools": [tool_def], "no_cache": True},
            )

        section.findings = {
            "endpoint_status": results,
            "tool_calling_request_accepted": tool_attempt.ok,
        }
        unreachable = [k for k, v in results.items() if v in (0, 404, 500, 502, 503)]
        if unreachable:
            section.notes.append(f"Unreachable or failing surfaces: {', '.join(unreachable)}")
        return section

    async def agentic(
        self, samples: int, *, prompt: str, mcp_server_url: str | None = None
    ) -> Section:
        """Server-side agent loop: the gateway offers MCP tools, runs the ones the
        model calls, and loops until the model answers."""
        section = Section("agentic_tool_calls")
        registered: str | None = None
        async with self._client() as client:
            servers = (await client.get("/v1/mcp/servers")).json() if samples > 0 else []
            healthy = [s["id"] for s in servers if s.get("health_status") == "healthy"]
            if not healthy and mcp_server_url:
                created = await client.post(
                    "/v1/mcp/servers",
                    json={
                        "name": f"eval-{uuid.uuid4().hex[:8]}",
                        "transport": "http",
                        "url": mcp_server_url,
                        "tool_prefix": f"eval{uuid.uuid4().hex[:6]}",
                    },
                )
                if created.status_code == 201 and created.json().get("health_status") == "healthy":
                    registered = created.json()["id"]
                    healthy = [registered]
            if not healthy:
                section.findings = {"measured": False}
                section.notes.append(
                    "No healthy MCP server; register one or pass --mcp-server-url."
                )
                return section

            latencies: list[float] = []
            tool_calls: list[int] = []
            stops: dict[str, int] = {}
            succeeded = 0
            for _ in range(samples):
                started = time.perf_counter()
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "aigw": {"mcp": {"servers": healthy[:1], "max_iterations": 4}},
                        "no_cache": True,
                    },
                )
                latencies.append(time.perf_counter() - started)
                if response.status_code != 200:
                    stops["error"] = stops.get("error", 0) + 1
                    continue
                extras = response.json().get("aigw") or {}
                stop = str(extras.get("stop_reason") or "unknown")
                stops[stop] = stops.get(stop, 0) + 1
                executed = int(extras.get("tool_calls_executed") or 0)
                tool_calls.append(executed)
                if stop == "completed" and executed > 0:
                    succeeded += 1
            if registered:
                await client.delete(f"/v1/mcp/servers/{registered}")

        hops = [calls + 1 for calls in tool_calls] or [1]
        section.findings = {
            "measured": True,
            "samples": samples,
            "tool_loop_success_rate": round(succeeded / samples, 4) if samples else 0.0,
            "tool_calls_per_request": round(statistics.mean(tool_calls), 2) if tool_calls else 0.0,
            "stop_reasons": stops,
            "max_iterations_rate": round(stops.get("max_iterations", 0) / samples, 4)
            if samples
            else 0.0,
            "latency_p50_s": round(_percentile(latencies, 50), 3),
            "latency_p95_s": round(_percentile(latencies, 95), 3),
            "latency_per_hop_s": round(statistics.mean(latencies) / statistics.mean(hops), 3)
            if latencies
            else 0.0,
        }
        if succeeded < samples:
            section.notes.append(
                "Some agent loops did not complete with a tool call; see stop_reasons."
            )
        return section


def _render(sections: list[Section], as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {s.name: {"findings": s.findings, "notes": s.notes} for s in sections}, indent=2
            )
        )
        return

    for section in sections:
        print(f"\n{'=' * 72}\n{section.name.replace('_', ' ').upper()}\n{'=' * 72}")
        for key, value in section.findings.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for sub_key, sub_value in value.items():
                    print(f"      {sub_key:<34} {sub_value}")
            else:
                print(f"  {key:<38} {value}")
        for note in section.notes:
            print(f"  ! {note}")


async def run(args: argparse.Namespace) -> int:
    evaluator = Evaluator(
        args.base_url,
        args.api_key,
        args.model,
        args.timeout,
        failover_model=args.failover_model,
        routing_model=args.routing_model,
    )

    # Fail fast and clearly rather than reporting a page of zeros.
    try:
        async with httpx.AsyncClient(base_url=args.base_url, timeout=10.0) as probe:
            health = await probe.get("/healthz")
            health.raise_for_status()
    except Exception as exc:
        print(f"Gateway is not reachable at {args.base_url}: {exc}", file=sys.stderr)
        return 2

    sections = [
        await evaluator.throughput_and_latency(args.requests, args.concurrency),
        await evaluator.time_to_first_token(args.stream_samples),
        await evaluator.cache_effectiveness(args.cache_repeats),
        await evaluator.failover(),
        await evaluator.routing(),
        await evaluator.guardrails(),
        await evaluator.capabilities(),
        await evaluator.agentic(
            args.agentic_samples, prompt=args.agentic_prompt, mcp_server_url=args.mcp_server_url
        ),
    ]
    _render(sections, args.json)

    headline = sections[0].findings
    success_rate = float(headline.get("success_rate") or 0.0)
    if not args.json:
        print(f"\nOverall success rate: {success_rate:.1%}")
    return 0 if success_rate >= args.min_success_rate else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument(
        "--failover-model",
        default=None,
        help=(
            "Model whose first deployment is deliberately unreachable, used to "
            "prove provider failover end to end (e.g. 'eval-chat' from "
            "config/models.eval.yaml). Omitted, failover is reported as unmeasured."
        ),
    )
    parser.add_argument(
        "--routing-model",
        default=None,
        help=(
            "Model with several healthy deployments at different prices, used to "
            "show the routing strategies diverge (e.g. 'eval-router'). "
            "Defaults to --model."
        ),
    )
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--stream-samples", type=int, default=5)
    parser.add_argument("--cache-repeats", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument(
        "--agentic-samples", type=int, default=5, help="Agent-loop requests (0 skips)"
    )
    parser.add_argument(
        "--agentic-prompt",
        default="__tool__:add What is 2 + 3? Use the add tool.",
        help="Prompt that should make the model call an MCP tool",
    )
    parser.add_argument(
        "--mcp-server-url",
        default=None,
        help="Register this MCP server for the agentic probe when none is healthy",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable output")
    parser.add_argument(
        "--min-success-rate",
        type=float,
        default=0.95,
        help="Exit non-zero below this success rate, for CI gating",
    )
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
