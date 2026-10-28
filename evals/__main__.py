"""``uv run python -m evals <dataset.yaml> [--out DIR]``

Writes ``<out>/eval.json`` (the summary, which ``scripts/bench_compare.py``
reads as ``eval.*`` metrics) and ``<out>/eval_cases.json`` (every case with its
answer, scores, and the judge's reasons). Exit code 1 when any case fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from evals.judge import GatewayJudge
from evals.runner import load_dataset, run
from evals.targets import build_target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=Path("bench/results/eval"))
    parser.add_argument(
        "--base-url", default=os.environ.get("AIGW_BASE_URL", "http://localhost:4000")
    )
    parser.add_argument("--api-key", default=os.environ.get("AIGW_API_KEY", ""))
    parser.add_argument("--model", help="Model to evaluate (overrides the dataset's target)")
    parser.add_argument("--judge-model", default=os.environ.get("EVAL_JUDGE_MODEL"))
    args = parser.parse_args(argv)

    if not args.api_key:
        parser.error("no gateway key: pass --api-key or set AIGW_API_KEY")
    dataset = load_dataset(args.dataset)
    spec = dict(dataset.get("target") or {})
    if args.model:
        spec["model"] = args.model
    model = spec.get("model", "")
    judge_model = args.judge_model or (dataset.get("judge") or {}).get("model") or model
    if not judge_model:
        parser.error("no judge model: set judge.model, --judge-model, or EVAL_JUDGE_MODEL")
    target = build_target(spec, base_url=args.base_url, api_key=args.api_key, model=model)
    judge = GatewayJudge(judge_model, base_url=args.base_url, api_key=args.api_key)

    summary, results = run(dataset, target, judge)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "eval.json").write_text(json.dumps(summary, indent=2))
    (args.out / "eval_cases.json").write_text(json.dumps(results, indent=2))
    for record in results:
        mark = "ok  " if record["passed"] else "FAIL"
        scores = ", ".join(f"{k}={v['score']:.2f}" for k, v in record["metrics"].items())
        print(f"  {mark} {record['id']}: {record['error'] or scores}")
    print(f"\n{summary['passed']}/{summary['cases']} cases passed — {args.out / 'eval.json'}")
    return 0 if summary["passed"] == summary["cases"] else 1


if __name__ == "__main__":
    sys.exit(main())
