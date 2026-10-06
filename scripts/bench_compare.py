"""Compare two benchmark runs and fail on regressions.

A run is a folder written by ``scripts/bench.py``: one JSON file per tool
(``evaluate.json``, ``loadtest.json``, ``agent_compat.json``, ``rag_eval.json``).
Each file is flattened into dotted metric names prefixed with the tool, such as
``loadtest.latency_p95_ms`` or ``rag_eval.vector.recall_at_5``.

``bench/thresholds.yaml`` says which metrics matter and how far each may move
before the comparison fails::

    metrics:
      loadtest.latency_p95_ms: {better: lower, max_increase_pct: 25}
      loadtest.success_rate:   {better: higher, max_drop: 0.01, min: 0.99}
      agent_compat.pass_rate:  {min: 1.0}

- ``max_increase_pct`` / ``max_drop_pct``: relative change allowed against the baseline.
- ``max_increase`` / ``max_drop``: absolute change allowed against the baseline.
- ``min`` / ``max``: absolute bounds on the current value, baseline or not.
- ``equals``: the current value must equal this (for booleans such as failover).

Usage::

    uv run python scripts/bench_compare.py bench/baseline bench/results/<run>

Exit status is 1 when any metric regresses or a gated metric is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

TOOLS = ("evaluate", "loadtest", "agent_compat", "rag_eval")


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten nested JSON into dotted keys; evaluate.py's "findings" level is dropped."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key == "notes":
                continue
            name = prefix if key == "findings" else f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(item, name))
        return out
    return {prefix: value}


def load_run(folder: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for tool in TOOLS:
        path = folder / f"{tool}.json"
        if path.exists():
            metrics.update(flatten(json.loads(path.read_text()), tool))
    return metrics


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def check(rule: dict[str, Any], baseline: Any, current: Any) -> list[str]:
    """Return the reasons ``current`` violates ``rule`` (empty when it passes)."""
    problems: list[str] = []
    if "equals" in rule and current != rule["equals"]:
        problems.append(f"expected {rule['equals']!r}")
    now = _number(current)
    if now is None:
        return problems
    if "min" in rule and now < float(rule["min"]):
        problems.append(f"below minimum {rule['min']}")
    if "max" in rule and now > float(rule["max"]):
        problems.append(f"above maximum {rule['max']}")
    before = _number(baseline)
    if before is None:
        return problems
    delta = now - before
    pct = (delta / abs(before) * 100) if before else (0.0 if delta == 0 else float("inf"))
    if "max_increase" in rule and delta > float(rule["max_increase"]):
        problems.append(f"rose {delta:+.4g} (allowed +{rule['max_increase']})")
    if "max_drop" in rule and -delta > float(rule["max_drop"]):
        problems.append(f"fell {delta:+.4g} (allowed -{rule['max_drop']})")
    if "max_increase_pct" in rule and pct > float(rule["max_increase_pct"]):
        problems.append(f"rose {pct:+.1f}% (allowed +{rule['max_increase_pct']}%)")
    if "max_drop_pct" in rule and -pct > float(rule["max_drop_pct"]):
        problems.append(f"fell {pct:+.1f}% (allowed -{rule['max_drop_pct']}%)")
    return problems


def compare(
    baseline: dict[str, Any], current: dict[str, Any], rules: dict[str, dict[str, Any]]
) -> tuple[list[tuple[str, Any, Any, str]], bool]:
    rows: list[tuple[str, Any, Any, str]] = []
    failed = False
    for name, rule in rules.items():
        before, now = baseline.get(name), current.get(name)
        if name not in current:
            if name in baseline or rule.get("required", True):
                rows.append((name, before, None, "FAIL: missing from this run"))
                failed = True
            continue
        problems = check(rule, before, now)
        failed = failed or bool(problems)
        rows.append((name, before, now, "FAIL: " + "; ".join(problems) if problems else "ok"))
    return rows, failed


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def render(rows: list[tuple[str, Any, Any, str]]) -> str:
    lines = ["| Metric | Baseline | Current | Result |", "|---|---|---|---|"]
    lines += [f"| `{n}` | {_fmt(b)} | {_fmt(c)} | {r} |" for n, b, c, r in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("baseline", type=Path, help="Baseline run folder")
    parser.add_argument("current", type=Path, help="Run folder to check")
    parser.add_argument("--thresholds", type=Path, default=Path("bench/thresholds.yaml"))
    args = parser.parse_args(argv)

    rules = (yaml.safe_load(args.thresholds.read_text()) or {}).get("metrics", {})
    rows, failed = compare(load_run(args.baseline), load_run(args.current), rules)
    table = render(rows)
    print(table)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as handle:
            handle.write(f"## Benchmark comparison\n\n{table}\n")
    print("\nREGRESSION" if failed else "\nNo regressions")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
