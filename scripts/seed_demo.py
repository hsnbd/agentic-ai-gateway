"""Seed the console with realistic demo traffic.

Writes demo teams and virtual keys, then 60 days of request logs, the matching
hourly usage rollups, and guardrail violations, so every console page (the
dashboard, usage, logs, keys, teams) has believable data to show.

The traffic is shaped like a real deployment: a daily cycle in local business
hours, quieter weekends, a nightly batch job, steady growth, cache hits that
cost nothing, and two provider incidents with failover. Rollups are built from
the same rows as the logs, so totals agree everywhere.

Every row hangs off a demo team or key marked ``{"demo_seed": true}`` in its
metadata; ``--reset`` removes exactly those and leaves real traffic alone.

Usage:

    uv run --env-file .env python scripts/seed_demo.py           # seed once
    uv run --env-file .env python scripts/seed_demo.py --reset   # replace demo data
    uv run --env-file .env python scripts/seed_demo.py --remove  # remove demo data only
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import secrets
import sys
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import delete, insert, select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.auth.keys import hash_key  # noqa: E402
from app.config.settings import get_settings  # noqa: E402
from app.db.models import (  # noqa: E402
    GuardrailViolation,
    RequestLog,
    Team,
    UsageRollup,
    VirtualKey,
)
from app.db.session import Database  # noqa: E402

DEMO_MARK = "demo_seed"
#: Local business hours drive the daily cycle (the team works in UTC+6).
LOCAL_TZ = timezone(timedelta(hours=6))
SUCCESS = "success"

# ---------------------------------------------------------------- model behaviour
#: Fixed overhead and per-output-token time (ms) for each public model name.
LATENCY: dict[str, tuple[float, float]] = {
    "gpt-4o": (420, 11.0),
    "gpt-4o-mini": (260, 7.5),
    "claude-sonnet-4": (650, 14.0),
    "claude-haiku-3-5": (320, 7.0),
    "gemini-2-flash": (240, 4.5),
    "llama3.2": (180, 24.0),
    "text-embedding-3-small": (70, 0.0),
    "nomic-embed-text": (35, 0.0),
}
#: Where each model fails over to when its provider is degraded.
FALLBACK = {"gpt-4o": "claude-sonnet-4", "gpt-4o-mini": "claude-haiku-3-5"}
EMBEDDING_MODELS = {"text-embedding-3-small", "nomic-embed-text"}

TEAMS = [
    ("Customer Support", "Support copilot and ticket triage.", 60),
    ("Knowledge", "Docs search and RAG answers for the help centre.", 8),
    ("Engineering", "Coding agents and internal developer tools.", 350),
    ("Data Science", "Batch enrichment, evals, and analytics.", 60),
    ("Growth", "Marketing copy and the mobile app assistant.", 40),
]


@dataclass
class KeyProfile:
    name: str
    team: str
    weight: float
    models: dict[str, float]
    prompt_median: float
    completion_median: float
    cache_rate: float = 0.0
    stream_rate: float = 0.6
    pattern: str = "business"  # business | night | flat
    weekend: float = 0.45
    growth: float = 0.6  # traffic at the start relative to now
    tools: float = 0.0
    rag: bool = False
    tags: list[str] = field(default_factory=list)
    end_user: str | None = None
    budget: float | None = None
    rpm: int | None = None


PROFILES = [
    KeyProfile(
        "support-copilot-prod",
        "Customer Support",
        0.30,
        {"gpt-4o-mini": 0.55, "claude-haiku-3-5": 0.25, "gpt-4o": 0.20},
        1800,
        260,
        cache_rate=0.22,
        stream_rate=0.85,
        tags=["support", "prod"],
        end_user="cust_{:05d}",
        budget=50,
        rpm=600,
    ),
    KeyProfile(
        "docs-search-rag",
        "Knowledge",
        0.18,
        {
            "text-embedding-3-small": 0.30,
            "nomic-embed-text": 0.15,
            "gemini-2-flash": 0.35,
            "gpt-4o-mini": 0.20,
        },
        3200,
        320,
        cache_rate=0.30,
        stream_rate=0.5,
        rag=True,
        tags=["rag", "prod"],
        end_user="visitor_{:06d}",
        budget=5,
    ),
    KeyProfile(
        "coding-agent",
        "Engineering",
        0.11,
        {"claude-sonnet-4": 0.75, "gpt-4o": 0.25},
        11000,
        850,
        cache_rate=0.03,
        stream_rate=0.95,
        weekend=0.25,
        tools=0.7,
        tags=["agent", "coding"],
        end_user="dev_{:03d}",
        budget=300,
    ),
    KeyProfile(
        "analytics-batch",
        "Data Science",
        0.10,
        {"gemini-2-flash": 0.6, "gpt-4o-mini": 0.4},
        1400,
        180,
        stream_rate=0.0,
        pattern="night",
        weekend=0.9,
        growth=0.8,
        tags=["batch"],
    ),
    KeyProfile(
        "marketing-copy",
        "Growth",
        0.05,
        {"gpt-4o": 0.5, "claude-sonnet-4": 0.5},
        900,
        650,
        cache_rate=0.05,
        weekend=0.1,
        tags=["marketing"],
        end_user="mkt_{:02d}",
        budget=30,
    ),
    KeyProfile(
        "internal-chat",
        "Engineering",
        0.12,
        {"llama3.2": 0.5, "gpt-4o-mini": 0.3, "claude-haiku-3-5": 0.2},
        700,
        300,
        cache_rate=0.12,
        stream_rate=0.9,
        weekend=0.2,
        tags=["internal"],
        end_user="emp_{:03d}",
    ),
    KeyProfile(
        "eval-runner",
        "Data Science",
        0.05,
        {"gpt-4o": 0.3, "claude-sonnet-4": 0.3, "gpt-4o-mini": 0.2, "gemini-2-flash": 0.2},
        2400,
        400,
        stream_rate=0.0,
        pattern="night",
        weekend=1.0,
        growth=0.9,
        tags=["eval", "ci"],
    ),
    KeyProfile(
        "mobile-app-beta",
        "Growth",
        0.09,
        {"gpt-4o-mini": 0.8, "gemini-2-flash": 0.2},
        600,
        180,
        cache_rate=0.15,
        stream_rate=0.9,
        pattern="flat",
        weekend=1.1,
        growth=0.15,
        tags=["mobile", "beta"],
        end_user="app_{:06d}",
        budget=2,
        rpm=300,
    ),
]


def load_models() -> dict[str, dict[str, Any]]:
    """Public model name -> provider, upstream model, and pricing, from config/models.yaml."""
    raw = yaml.safe_load((ROOT / "config" / "models.yaml").read_text())
    models: dict[str, dict[str, Any]] = {}
    for entry in raw["model_list"]:
        models.setdefault(
            entry["model_name"],
            {
                "provider": entry["params"]["provider"],
                "upstream": entry["params"]["model"],
                "pricing": entry.get("pricing", {}),
            },
        )
    return models


# ---------------------------------------------------------------- traffic shape
def hour_shape(pattern: str, local_hour: int) -> float:
    if pattern == "night":  # batch jobs at 01:00-05:00
        return 6.0 if 1 <= local_hour <= 4 else 0.25
    if pattern == "flat":  # consumer app: evenings peak
        return (
            0.5
            + 0.5 * math.sin((local_hour - 13) / 24 * 2 * math.pi)
            + (0.6 if 19 <= local_hour <= 23 else 0)
        )
    # Business hours: ramp from 08:00, lunch dip, tail into the evening.
    curve = math.exp(-((local_hour - 11.5) ** 2) / 8) + 0.9 * math.exp(
        -((local_hour - 16) ** 2) / 6
    )
    return 0.06 + curve


def poisson(rng: random.Random, lam: float) -> int:
    if lam <= 0:
        return 0
    if lam > 50:
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


def weighted(rng: random.Random, options: dict[str, float]) -> str:
    return rng.choices(list(options), weights=list(options.values()))[0]


@dataclass
class Incident:
    start: datetime
    end: datetime
    provider: str
    fail_rate: float
    error_code: str
    status_code: int
    message: str


def incidents(now: datetime) -> list[Incident]:
    day = now.astimezone(LOCAL_TZ).replace(minute=0, second=0, microsecond=0)
    openai = (day - timedelta(days=3)).replace(hour=14).astimezone(UTC)
    ollama = (day - timedelta(days=9)).replace(hour=10).astimezone(UTC)
    gemini = (day - timedelta(days=19)).replace(hour=2).astimezone(UTC)
    return [
        Incident(
            openai,
            openai + timedelta(minutes=150),
            "openai",
            0.38,
            "provider_rate_limit",
            429,
            "Rate limit reached for requests in organization (tokens per min).",
        ),
        Incident(
            ollama,
            ollama + timedelta(minutes=95),
            "ollama",
            0.92,
            "provider_unavailable",
            503,
            "connection refused: ollama:11434",
        ),
        Incident(
            gemini,
            gemini + timedelta(minutes=50),
            "gemini",
            0.25,
            "provider_overloaded",
            503,
            "The model is overloaded. Please try again later.",
        ),
    ]


BACKGROUND_ERRORS = [
    # (probability, code, status, message)
    (0.0050, "provider_timeout", 504, "upstream did not respond within 60s"),
    (0.0025, "rate_limit_exceeded", 429, "key exceeded its rpm limit"),
    (0.0015, "context_length_exceeded", 400, "prompt exceeds the model's context window"),
    (0.0010, "provider_error", 502, "upstream returned 500 Internal Server Error"),
    (0.0006, "budget_exceeded", 402, "key budget exhausted for this period"),
]

PII_RULES = [
    ("redact-pii", "EMAIL", "jane….com"),
    ("redact-pii", "PHONE", "+880…4471"),
    ("redact-pii", "CREDIT_CARD", "4111…1111"),
]


# ---------------------------------------------------------------- generation
class Generator:
    def __init__(
        self,
        rng: random.Random,
        models: dict[str, dict[str, Any]],
        keys: dict[str, tuple[str, str]],
        now: datetime,
    ) -> None:
        self.rng = rng
        self.models = models
        self.keys = keys  # profile name -> (key id, team id)
        self.now = now
        self.incidents = incidents(now)
        self.logs: list[dict[str, Any]] = []
        self.violations: list[dict[str, Any]] = []
        self.rollups: dict[tuple[datetime, str, str, str], dict[str, Any]] = {}

    def cost(self, model: str, prompt: int, completion: int, cached: int) -> float:
        price = self.models[model]["pricing"]
        fresh = prompt - cached
        return (
            fresh * price.get("input_per_mtok", 0)
            + cached * price.get("cached_input_per_mtok", price.get("input_per_mtok", 0))
            + completion * price.get("output_per_mtok", 0)
        ) / 1_000_000

    def latency(self, model: str, completion: int) -> float:
        overhead, per_token = LATENCY[model]
        jitter = self.rng.lognormvariate(0, 0.35)
        return (overhead + per_token * completion) * jitter

    def incident_for(self, provider: str, at: datetime) -> Incident | None:
        for incident in self.incidents:
            if incident.provider == provider and incident.start <= at < incident.end:
                return incident
        return None

    def run(self, days: int, daily_requests: float) -> None:
        start = (self.now - timedelta(days=days)).replace(minute=0, second=0, microsecond=0)
        hour = start
        span = (self.now - start).total_seconds()
        while hour <= self.now:
            local = hour.astimezone(LOCAL_TZ)
            progress = (hour - start).total_seconds() / span
            weekend = local.weekday() >= 4  # Friday and Saturday off in Bangladesh
            for profile in PROFILES:
                shape = hour_shape(profile.pattern, local.hour)
                norm = sum(hour_shape(profile.pattern, h) for h in range(24))
                growth = profile.growth + (1 - profile.growth) * progress
                lam = daily_requests * profile.weight * shape / norm * growth
                lam *= profile.weekend if weekend else 1.0
                lam *= self.rng.lognormvariate(0, 0.18)
                for _ in range(poisson(self.rng, lam)):
                    at = hour + timedelta(seconds=self.rng.uniform(0, 3600))
                    if at <= self.now:
                        self.request(profile, at)
            hour += timedelta(hours=1)

    def request(self, profile: KeyProfile, at: datetime) -> None:
        rng = self.rng
        key_id, team_id = self.keys[profile.name]
        model = weighted(rng, profile.models)
        info = self.models[model]
        provider, served_model, deployment = info["provider"], model, f"{info['provider']}/{model}"
        embedding = model in EMBEDDING_MODELS
        route = (
            "/v1/embeddings"
            if embedding
            else (
                "/v1/messages"
                if provider == "anthropic" and profile.name == "coding-agent"
                else "/v1/chat/completions"
            )
        )
        dialect = "anthropic" if route == "/v1/messages" else "openai"

        prompt = max(
            8,
            int(
                rng.lognormvariate(
                    math.log(profile.prompt_median * (0.12 if embedding else 1)), 0.7
                )
            ),
        )
        completion = (
            0
            if embedding
            else max(4, int(rng.lognormvariate(math.log(profile.completion_median), 0.6)))
        )
        cached = int(prompt * rng.uniform(0.3, 0.8)) if not embedding and rng.random() < 0.35 else 0
        stream = not embedding and rng.random() < profile.stream_rate
        tool_calls = min(8, poisson(rng, 2.2)) if rng.random() < profile.tools else 0

        status, error_code, status_code, error_message = SUCCESS, None, None, None
        attempts: list[dict[str, Any]] = []
        fallback, reason, strategy = False, "priority 0 (lowest wins)", "priority"
        cache_hit = not embedding and rng.random() < profile.cache_rate
        cost, saved, latency = 0.0, 0.0, 0.0
        flagged, guardrails = False, {}

        timings: dict[str, Any] = {
            "auth": round(rng.uniform(0.8, 4.5), 3),
            "guardrails_input": round(rng.uniform(0.05, 0.6), 3),
            "cache": round(rng.uniform(4, 18), 3) if not embedding else 0.003,
        }
        if profile.rag and not embedding:
            timings["rag"] = round(rng.uniform(35, 140), 3)

        # Guardrails: PII is redacted and the request continues; oversized input is blocked.
        if profile.end_user and rng.random() < 0.018:
            rule, entity, excerpt = rng.choice(PII_RULES)
            flagged = True
            guardrails = {"input": [{"rule": rule, "action": "redact", "entities": {entity: 1}}]}
            self.violation(
                at, key_id, "redact-pii", "redact", "high", excerpt, {"entities": {entity: 1}}
            )
        elif rng.random() < 0.003:
            status, error_code, status_code = "error", "guardrail_violation", 400
            error_message = "input blocked by guardrail input-length"
            chars = rng.randint(20_100, 48_000)
            self.violation(
                at,
                key_id,
                "input-length",
                "block",
                "medium",
                "…",
                {"exceeded": {"characters": chars}, "max_chars": 20000, "max_messages": 100},
            )

        if status == SUCCESS and cache_hit:
            latency = rng.uniform(9, 45) + sum(v for v in timings.values() if isinstance(v, float))
            saved = self.cost(model, prompt, completion, 0)
            strategy, reason = None, None
        elif status == SUCCESS:
            incident = self.incident_for(provider, at)
            failed_first = incident is not None and rng.random() < incident.fail_rate
            if failed_first:
                assert incident is not None
                wasted = rng.uniform(150, 1800)
                attempts.append(
                    {
                        "error": incident.error_code,
                        "outcome": "error",
                        "provider": provider,
                        "latency_ms": round(wasted, 3),
                        "deployment_id": deployment,
                    }
                )
                backup = FALLBACK.get(model)
                if backup and rng.random() < 0.9:
                    fallback = True
                    served_model = backup
                    provider = self.models[backup]["provider"]
                    reason = f"fallback from {deployment} after {incident.error_code}"
                    deployment = f"{provider}/{backup}"
                    exec_ms = self.latency(backup, completion)
                    attempts.append(
                        {
                            "error": None,
                            "outcome": "success",
                            "provider": provider,
                            "latency_ms": round(exec_ms, 3),
                            "deployment_id": deployment,
                        }
                    )
                    latency = wasted + exec_ms
                    cost = self.cost(backup, prompt, completion, cached)
                else:
                    status, status_code = "error", incident.status_code
                    error_code = "all_providers_failed" if backup else incident.error_code
                    error_message = incident.message
                    latency = wasted
            else:
                for probability, code, http, message in BACKGROUND_ERRORS:
                    if rng.random() < probability:
                        status, error_code, status_code, error_message = (
                            "error",
                            code,
                            http,
                            message,
                        )
                        break
                if status == SUCCESS:
                    latency = self.latency(model, completion)
                    if tool_calls:
                        latency += tool_calls * rng.uniform(250, 1400)
                    cost = self.cost(model, prompt, completion, cached)
                else:
                    latency = (
                        rng.uniform(800, 60_000)
                        if error_code == "provider_timeout"
                        else rng.uniform(5, 300)
                    )
                attempts.append(
                    {
                        "error": error_code if status != SUCCESS else None,
                        "outcome": status,
                        "provider": provider,
                        "latency_ms": round(latency, 3),
                        "deployment_id": deployment,
                    }
                )
            timings["execute"] = round(latency, 3)
            timings["attempts"] = attempts
            if status == SUCCESS:
                timings["guardrails_output"] = round(rng.uniform(0.02, 0.4), 3)
                timings["cache_write"] = round(rng.uniform(0.01, 2.5), 3)
            latency += timings["auth"] + timings["cache"] + timings.get("rag", 0)

        succeeded = status == SUCCESS
        if not succeeded:
            completion, cached, tool_calls = 0, 0, 0
            if error_code in {
                "guardrail_violation",
                "rate_limit_exceeded",
                "budget_exceeded",
                "context_length_exceeded",
            }:
                prompt = 0 if error_code != "context_length_exceeded" else prompt
                strategy, reason, served_model = None, None, None
        total = prompt + completion if succeeded else 0
        request_id = f"req_{uuid.uuid4().hex[:24]}"
        end_user = None
        if profile.end_user:
            pool = {"cust": 4000, "visitor": 90000, "dev": 45, "mkt": 8, "emp": 160, "app": 25000}
            prefix = profile.end_user.split("_")[0]
            end_user = profile.end_user.format(rng.randint(1, pool[prefix]))

        self.logs.append(
            {
                "id": str(uuid.uuid4()),
                "request_id": request_id,
                "created_at": at,
                "virtual_key_id": key_id,
                "team_id": team_id,
                "end_user": end_user,
                "model": model,
                "resolved_model": self.models[served_model]["upstream"]
                if succeeded and served_model
                else None,
                "provider": provider,
                "deployment_id": deployment if strategy else None,
                "route": route,
                "dialect": dialect,
                "status": status,
                "status_code": status_code,
                "error_code": error_code,
                "error_message": error_message,
                "prompt_tokens": prompt if succeeded else 0,
                "completion_tokens": completion,
                "total_tokens": total,
                "cached_tokens": cached,
                "cost_usd": cost,
                "latency_ms": latency,
                "time_to_first_token_ms": (
                    min(latency, LATENCY[served_model][0] * rng.uniform(0.6, 1.6))
                    if stream and succeeded and not cache_hit and served_model
                    else None
                ),
                "stage_timings": timings,
                "attempt_count": max(1, len(attempts)),
                "fallback_used": fallback,
                "routing_strategy": strategy,
                "routing_reason": reason,
                "cache_hit": cache_hit and succeeded,
                "cache_similarity": round(rng.uniform(0.93, 1.0), 4)
                if cache_hit and succeeded
                else None,
                "guardrail_flagged": flagged,
                "guardrail_results": guardrails,
                "stream": stream,
                "tool_calls_count": tool_calls,
                "trace_id": uuid.uuid4().hex,
                "tags": profile.tags,
                "request_body": None,
                "response_body": None,
            }
        )
        self.violations_attach(request_id)

        bucket = at.replace(minute=0, second=0, microsecond=0)
        slot = (bucket, key_id, model, provider)
        roll = self.rollups.setdefault(
            slot,
            {
                "id": str(uuid.uuid4()),
                "bucket": bucket,
                "virtual_key_id": key_id,
                "team_id": team_id,
                "model": model,
                "provider": provider,
                "request_count": 0,
                "success_count": 0,
                "error_count": 0,
                "cache_hit_count": 0,
                "fallback_count": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost_usd": 0.0,
                "cost_saved_usd": 0.0,
                "total_latency_ms": 0.0,
            },
        )
        roll["request_count"] += 1
        roll["success_count"] += int(succeeded)
        roll["error_count"] += int(not succeeded)
        roll["cache_hit_count"] += int(cache_hit and succeeded)
        roll["fallback_count"] += int(fallback)
        roll["prompt_tokens"] += prompt if succeeded else 0
        roll["completion_tokens"] += completion
        roll["total_tokens"] += total
        roll["cost_usd"] += cost
        roll["cost_saved_usd"] += saved
        roll["total_latency_ms"] += latency

    def violation(
        self,
        at: datetime,
        key_id: str,
        rule: str,
        action: str,
        severity: str,
        excerpt: str,
        details: dict[str, Any],
    ) -> None:
        self.violations.append(
            {
                "id": str(uuid.uuid4()),
                "created_at": at,
                "request_id": "",
                "virtual_key_id": key_id,
                "policy": "default",
                "rule": rule,
                "phase": "input",
                "action": action,
                "severity": severity,
                "match_count": 1,
                "excerpt": excerpt,
                "details": details,
            }
        )

    def violations_attach(self, request_id: str) -> None:
        if self.violations and not self.violations[-1]["request_id"]:
            self.violations[-1]["request_id"] = request_id


# ---------------------------------------------------------------- persistence
async def demo_ids(session: Any) -> tuple[list[str], list[str]]:
    teams = [
        t.id
        for t in (await session.scalars(select(Team))).all()
        if (t.metadata_ or {}).get(DEMO_MARK)
    ]
    keys = [
        k.id
        for k in (await session.scalars(select(VirtualKey))).all()
        if (k.metadata_ or {}).get(DEMO_MARK)
    ]
    return teams, keys


async def remove(db: Database) -> int:
    async with db.session() as session:
        teams, keys = await demo_ids(session)
        if keys:
            for model in (RequestLog, UsageRollup, GuardrailViolation):
                await session.execute(delete(model).where(model.virtual_key_id.in_(keys)))
            await session.execute(delete(VirtualKey).where(VirtualKey.id.in_(keys)))
        if teams:
            await session.execute(delete(Team).where(Team.id.in_(teams)))
        return len(keys)


async def seed(db: Database, args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    models = load_models()
    now = datetime.now(UTC)
    month_start = now.astimezone(LOCAL_TZ).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    next_month = (month_start + timedelta(days=32)).replace(day=1)

    team_ids = {name: str(uuid.uuid4()) for name, _, _ in TEAMS}
    key_ids = {p.name: (str(uuid.uuid4()), team_ids[p.team]) for p in PROFILES}

    generator = Generator(rng, models, key_ids, now)
    generator.run(args.days, args.daily)

    month_spend: dict[str, float] = defaultdict(float)
    last_used: dict[str, datetime] = {}
    for row in generator.logs:
        if row["created_at"] >= month_start:
            month_spend[row["virtual_key_id"]] += row["cost_usd"]
        last_used[row["virtual_key_id"]] = max(
            last_used.get(row["virtual_key_id"], row["created_at"]), row["created_at"]
        )

    async with db.session() as session:
        created = now - timedelta(days=args.days + 3)
        for name, description, budget in TEAMS:
            team_id = team_ids[name]
            spend = sum(
                month_spend[k] for p in PROFILES if p.team == name for k in [key_ids[p.name][0]]
            )
            session.add(
                Team(
                    id=team_id,
                    name=name,
                    description=description,
                    max_budget_usd=Decimal(budget),
                    spend_usd=Decimal(f"{spend:.6f}"),
                    budget_period="monthly",
                    budget_reset_at=next_month,
                    metadata_={DEMO_MARK: True},
                    created_at=created,
                    updated_at=now,
                )
            )
        await session.flush()
        for profile in PROFILES:
            key_id, team_id = key_ids[profile.name]
            raw = f"sk-aigw-{secrets.token_urlsafe(40)}"
            session.add(
                VirtualKey(
                    id=key_id,
                    key_hash=hash_key(raw),
                    key_prefix=raw[:16],
                    name=profile.name,
                    team_id=team_id,
                    max_budget_usd=Decimal(profile.budget) if profile.budget else None,
                    spend_usd=Decimal(f"{month_spend[key_id]:.6f}"),
                    budget_period="monthly",
                    budget_reset_at=next_month,
                    rpm_limit=profile.rpm,
                    allowed_models=sorted(profile.models),
                    is_active=True,
                    last_used_at=last_used.get(key_id),
                    metadata_={DEMO_MARK: True},
                    created_at=created + timedelta(days=rng.randint(0, 4)),
                    updated_at=now,
                )
            )
        await session.flush()

        for table, rows in (
            (RequestLog, generator.logs),
            (UsageRollup, list(generator.rollups.values())),
            (GuardrailViolation, generator.violations),
        ):
            for i in range(0, len(rows), args.batch):
                await session.execute(insert(table), rows[i : i + args.batch])

    requests = len(generator.logs)
    errors = sum(1 for r in generator.logs if r["status"] != SUCCESS)
    spend = sum(r["cost_usd"] for r in generator.logs)
    print(
        f"Seeded {len(TEAMS)} teams, {len(PROFILES)} keys, {requests:,} requests "
        f"({errors:,} errors, {sum(r['fallback_used'] for r in generator.logs):,} failovers), "
        f"{len(generator.rollups):,} rollups, {len(generator.violations):,} guardrail violations, "
        f"${spend:,.2f} spend over {args.days} days."
    )


async def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--days",
        type=int,
        default=60,
        help="history to generate (default 60, so 30d has a previous window)",
    )
    parser.add_argument(
        "--daily", type=float, default=2600, help="requests per weekday at today's volume"
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for a repeatable dataset"
    )
    parser.add_argument("--batch", type=int, default=2000)
    parser.add_argument("--reset", action="store_true", help="remove existing demo data first")
    parser.add_argument("--remove", action="store_true", help="remove demo data and exit")
    args = parser.parse_args()

    db = Database(get_settings())
    await db.startup()
    try:
        if args.remove or args.reset:
            removed = await remove(db)
            print(f"Removed demo data ({removed} keys and their traffic).")
            if args.remove:
                return 0
        else:
            async with db.session() as session:
                if (await demo_ids(session))[1]:
                    print("Demo data already exists; pass --reset to replace it.", file=sys.stderr)
                    return 1
        await seed(db, args)
    finally:
        await db.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
