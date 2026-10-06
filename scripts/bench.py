"""Run every benchmark against a gateway and save the results as one run.

Writes ``<out>/<UTC timestamp>/`` containing ``evaluate.json``,
``loadtest.json``, ``agent_compat.json``, ``rag_eval.json`` and ``meta.json``
(git commit, time, target, settings). With ``--baseline`` it then runs
``scripts/bench_compare.py`` against that folder and fails on regressions.

Defaults target the dockerised e2e stack (``make e2e-up``) and the evaluation
catalogue in ``config/models.eval.yaml``::

    uv run python scripts/bench.py --baseline bench/baseline
    uv run python scripts/bench.py --base-url https://gw.example.com --api-key sk-... \\
        --model gpt-4o-mini --failover-model "" --routing-model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _run(name: str, command: list[str], out: Path) -> dict[str, object]:
    print(f"▶ {name}", flush=True)
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, text=True)
    elapsed = round(time.monotonic() - started, 2)
    try:
        json.loads(result.stdout)
    except json.JSONDecodeError:
        print(f"  {name} produced no JSON (exit {result.returncode})\n{result.stderr[-2000:]}")
        return {"exit_code": result.returncode, "seconds": elapsed, "json": False}
    (out / f"{name}.json").write_text(result.stdout)
    status = "ok" if result.returncode == 0 else f"exit {result.returncode}"
    print(f"  {status} in {elapsed}s", flush=True)
    return {"exit_code": result.returncode, "seconds": elapsed, "json": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--base-url", default=os.environ.get("BENCH_BASE_URL", "http://localhost:18000")
    )
    parser.add_argument("--api-key", default=os.environ.get("BENCH_API_KEY", "sk-e2e-master-key"))
    parser.add_argument("--model", default="eval-router")
    parser.add_argument(
        "--failover-model", default="eval-chat", help='"" to skip the failover probe'
    )
    parser.add_argument("--routing-model", default="eval-router")
    parser.add_argument("--compat-model", default="eval-chat")
    parser.add_argument(
        "--mcp-server-url",
        default="http://fake-mcp:4200/mcp",
        help='MCP server (as the gateway sees it) for the agentic probe; "" to skip',
    )
    parser.add_argument("--requests", type=int, default=50)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--load-duration", type=float, default=30.0)
    parser.add_argument("--load-concurrency", type=int, default=10)
    parser.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=["evaluate", "loadtest", "agent_compat", "rag_eval"],
    )
    parser.add_argument("--out", type=Path, default=Path("bench/results"))
    parser.add_argument("--baseline", type=Path, help="Compare against this run folder")
    parser.add_argument("--thresholds", type=Path, default=Path("bench/thresholds.yaml"))
    args = parser.parse_args(argv)

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out = args.out / stamp
    out.mkdir(parents=True, exist_ok=True)
    target = ["--base-url", args.base_url, "--api-key", args.api_key]
    python = [sys.executable]

    evaluate = [
        *python,
        str(SCRIPTS / "evaluate.py"),
        *target,
        "--model",
        args.model,
        "--routing-model",
        args.routing_model,
        "--requests",
        str(args.requests),
        "--concurrency",
        str(args.concurrency),
        "--json",
    ]
    if args.failover_model:
        evaluate += ["--failover-model", args.failover_model]
    if args.mcp_server_url:
        evaluate += ["--mcp-server-url", args.mcp_server_url]
    commands = {
        "evaluate": evaluate,
        "loadtest": [
            *python,
            str(SCRIPTS / "loadtest.py"),
            *target,
            "--model",
            args.model,
            "--duration",
            str(args.load_duration),
            "--concurrency",
            str(args.load_concurrency),
            "--json",
        ],
        "agent_compat": [
            *python,
            str(SCRIPTS / "agent_compat.py"),
            *target,
            "--model",
            args.compat_model,
            "--json",
        ],
        "rag_eval": [*python, str(SCRIPTS / "rag_eval.py"), *target, "--json"],
    }

    tools = {}
    for name, command in commands.items():
        if name not in args.skip:
            tools[name] = _run(name, command, out)

    meta = {
        "started_at": stamp,
        "git_commit": _git_commit(),
        "base_url": args.base_url,
        "settings": {k: str(v) for k, v in vars(args).items() if k not in {"api_key", "skip"}},
        "tools": tools,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\nResults: {out}")

    failed = any(not tool["json"] for tool in tools.values())
    if args.baseline:
        from bench_compare import main as compare

        failed = (
            compare([str(args.baseline), str(out), "--thresholds", str(args.thresholds)]) != 0
            or failed
        )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.path.insert(0, str(SCRIPTS))
    sys.exit(main())
