"""Sustained load test for the gateway.

Separate from `scripts/evaluate.py` on purpose. The evaluator answers "is each
capability correct?" with small, targeted probes. This answers "what happens
when the gateway is under continuous pressure for a while?" — which surfaces a
different class of problem: connection-pool exhaustion, event-loop starvation,
cache and breaker behaviour at steady state, and latency drift as queues build.

It reports latency **percentiles over time**, not just an aggregate, because a
mean that stays flat while p99 climbs is the normal signature of a saturating
system and an aggregate number hides it.

Usage:

    uv run python scripts/loadtest.py \\
        --base-url http://localhost:4030 --api-key sk-... \\
        --model gpt-4o-mini --duration 60 --concurrency 20

Exit codes: 0 success, 1 threshold breached, 2 gateway unreachable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
import time
import uuid
from dataclasses import dataclass, field

import httpx

# Varied prompts keep the run honest: a single repeated prompt would be served
# from cache after the first request and measure almost nothing.
_TOPICS = [
    "photosynthesis",
    "the water cycle",
    "plate tectonics",
    "supply and demand",
    "the Doppler effect",
    "binary search",
    "the Krebs cycle",
    "cloud condensation",
]


@dataclass
class Result:
    ok: bool
    latency_s: float
    status: int
    cache_hit: bool = False
    error: str | None = None


@dataclass
class Bucket:
    """One reporting window."""

    second: int
    latencies: list[float] = field(default_factory=list)
    ok: int = 0
    failed: int = 0
    cache_hits: int = 0


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(round((pct / 100.0) * (len(ordered) - 1)), len(ordered) - 1)
    return ordered[index]


class LoadTest:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.results: list[Result] = []
        self.buckets: dict[int, Bucket] = {}
        self._started = 0.0
        self._stop = False

    def _record(self, result: Result) -> None:
        self.results.append(result)
        second = int(time.perf_counter() - self._started)
        bucket = self.buckets.setdefault(second, Bucket(second=second))
        bucket.latencies.append(result.latency_s)
        if result.ok:
            bucket.ok += 1
            if result.cache_hit:
                bucket.cache_hits += 1
        else:
            bucket.failed += 1

    async def _one(self, client: httpx.AsyncClient) -> Result:
        # A unique suffix on most requests prevents the cache from absorbing the
        # entire load; --cache-ratio controls how much of it is cacheable so the
        # test can model a realistic repeat rate.
        topic = random.choice(_TOPICS)
        if random.random() < self.args.cache_ratio:
            prompt = f"Explain {topic} in one sentence."
        else:
            prompt = f"Explain {topic} in one sentence. [{uuid.uuid4()}]"

        body = {
            "model": self.args.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.args.max_tokens,
        }
        started = time.perf_counter()
        try:
            response = await client.post("/v1/chat/completions", json=body)
        except Exception as exc:
            return Result(False, time.perf_counter() - started, 0, error=str(exc))

        latency = time.perf_counter() - started
        if response.status_code != 200:
            return Result(
                False, latency, response.status_code, error=response.text[:200]
            )
        return Result(
            True,
            latency,
            200,
            cache_hit=(response.headers.get("x-gateway-cache", "").lower() == "hit"),
        )

    async def _worker(self, client: httpx.AsyncClient) -> None:
        while not self._stop:
            self._record(await self._one(client))

    async def _reporter(self) -> None:
        """Print a line per window so drift is visible while the test runs."""
        if self.args.json:
            return
        last = 0
        header = (
            f"{'t(s)':>5} {'rps':>7} {'ok':>6} {'err':>5} "
            f"{'hit%':>6} {'p50ms':>7} {'p95ms':>8} {'p99ms':>8}"
        )
        print(header)
        print("-" * len(header))
        while not self._stop:
            await asyncio.sleep(self.args.interval)
            now = int(time.perf_counter() - self._started)
            window = [b for b in self.buckets.values() if last <= b.second < now]
            last = now
            if not window:
                continue
            latencies = [x for b in window for x in b.latencies]
            ok = sum(b.ok for b in window)
            failed = sum(b.failed for b in window)
            hits = sum(b.cache_hits for b in window)
            span = max(self.args.interval, 1)
            hit_pct = (hits / ok * 100.0) if ok else 0.0
            print(
                f"{now:>5} {(ok + failed) / span:>7.1f} {ok:>6} {failed:>5} "
                f"{hit_pct:>5.1f}% {_percentile(latencies, 50) * 1000:>7.1f} "
                f"{_percentile(latencies, 95) * 1000:>8.1f} "
                f"{_percentile(latencies, 99) * 1000:>8.1f}"
            )

    async def run(self) -> int:
        try:
            async with httpx.AsyncClient(base_url=self.args.base_url, timeout=10.0) as probe:
                (await probe.get("/healthz")).raise_for_status()
        except Exception as exc:
            print(f"Gateway is not reachable at {self.args.base_url}: {exc}", file=sys.stderr)
            return 2

        headers = {
            "Authorization": f"Bearer {self.args.api_key}",
            "Content-Type": "application/json",
        }
        limits = httpx.Limits(
            max_connections=self.args.concurrency * 2,
            max_keepalive_connections=self.args.concurrency,
        )

        self._started = time.perf_counter()
        async with httpx.AsyncClient(
            base_url=self.args.base_url,
            headers=headers,
            timeout=self.args.timeout,
            limits=limits,
        ) as client:
            reporter = asyncio.create_task(self._reporter())
            workers = [
                asyncio.create_task(self._worker(client))
                for _ in range(self.args.concurrency)
            ]
            await asyncio.sleep(self.args.duration)
            self._stop = True
            await asyncio.gather(*workers, return_exceptions=True)
            reporter.cancel()

        return self._summarise()

    def _summarise(self) -> int:
        elapsed = time.perf_counter() - self._started
        total = len(self.results)
        if not total:
            print("No requests completed.", file=sys.stderr)
            return 1

        ok = [r for r in self.results if r.ok]
        latencies = [r.latency_s for r in ok]
        success_rate = len(ok) / total
        hits = sum(1 for r in ok if r.cache_hit)

        # Comparing the first and last tenth shows whether latency drifted under
        # sustained load. A stable system holds; a saturating one climbs.
        drift: float | None = None
        if len(latencies) >= 20:
            slice_size = max(len(latencies) // 10, 1)
            head = statistics.mean(latencies[:slice_size])
            tail = statistics.mean(latencies[-slice_size:])
            drift = (tail - head) * 1000

        summary = {
            "duration_s": round(elapsed, 2),
            "concurrency": self.args.concurrency,
            "requests": total,
            "succeeded": len(ok),
            "failed": total - len(ok),
            "success_rate": round(success_rate, 4),
            "throughput_rps": round(total / elapsed, 2),
            "cache_hit_ratio": round(hits / len(ok), 4) if ok else 0.0,
            "latency_p50_ms": round(_percentile(latencies, 50) * 1000, 2),
            "latency_p95_ms": round(_percentile(latencies, 95) * 1000, 2),
            "latency_p99_ms": round(_percentile(latencies, 99) * 1000, 2),
            "latency_max_ms": round(max(latencies) * 1000, 2) if latencies else 0.0,
            "latency_drift_ms": round(drift, 2) if drift is not None else None,
        }

        errors: dict[str, int] = {}
        for result in self.results:
            if not result.ok:
                label = f"{result.status}: {(result.error or 'unknown')[:80]}"
                errors[label] = errors.get(label, 0) + 1
        summary["errors"] = errors

        if self.args.json:
            print(json.dumps(summary, indent=2))
        else:
            print("\n" + "=" * 64)
            print("LOAD TEST SUMMARY")
            print("=" * 64)
            for key, value in summary.items():
                if key != "errors":
                    print(f"  {key:<24} {value}")
            if errors:
                print("\n  errors:")
                for label, count in sorted(errors.items(), key=lambda kv: -kv[1]):
                    print(f"    {count:>6}  {label}")

        failures = []
        if success_rate < self.args.min_success_rate:
            failures.append(
                f"success rate {success_rate:.1%} below threshold "
                f"{self.args.min_success_rate:.1%}"
            )
        p95_ms = _percentile(latencies, 95) * 1000
        if self.args.max_p95_ms and p95_ms > self.args.max_p95_ms:
            failures.append(f"p95 {p95_ms:.0f}ms above threshold {self.args.max_p95_ms}ms")

        if failures:
            print("\nFAILED:", file=sys.stderr)
            for failure in failures:
                print(f"  - {failure}", file=sys.stderr)
            return 1
        if not self.args.json:
            print("\nPASSED")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Sustained load test for the gateway.")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--duration", type=float, default=30.0, help="Seconds to sustain load.")
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--interval", type=float, default=5.0, help="Reporting window seconds.")
    parser.add_argument(
        "--cache-ratio",
        type=float,
        default=0.3,
        help="Fraction of requests that are repeatable and therefore cacheable.",
    )
    parser.add_argument("--min-success-rate", type=float, default=0.99)
    parser.add_argument(
        "--max-p95-ms", type=float, default=0, help="Fail if p95 exceeds this. 0 disables."
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    try:
        return asyncio.run(LoadTest(args).run())
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
