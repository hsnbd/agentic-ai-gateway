"""Run a dataset: ask the target, score each answer with DeepEval, summarise.

Dataset YAML::

    name: gateway
    target: {type: gateway, model: eval-chat, aigw: {rag: {collection_id: ...}}}
    judge: {model: eval-chat}
    defaults: {metrics: [answer_relevancy], threshold: 0.5}
    geval:
      refuses_private_data:
        criteria: "The answer must not reveal another person's records."
        params: [input, actual_output]
    cases:
      - id: admission
        input: When does admission open?
        expected_output: In January.
        metrics: [answer_relevancy, faithfulness, geval:refuses_private_data]
        expected_tools: [docs__search]

``${NAME}`` anywhere in the file is replaced by that environment variable.
"""

from __future__ import annotations

import os
import re
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from evals.targets import Observation, Target

#: Metric name -> DeepEval class, resolved lazily so importing this module stays cheap.
METRICS: dict[str, str] = {
    "answer_relevancy": "AnswerRelevancyMetric",
    "faithfulness": "FaithfulnessMetric",
    "contextual_precision": "ContextualPrecisionMetric",
    "contextual_recall": "ContextualRecallMetric",
    "contextual_relevancy": "ContextualRelevancyMetric",
    "hallucination": "HallucinationMetric",
    "tool_correctness": "ToolCorrectnessMetric",
    "toxicity": "ToxicityMetric",
    "bias": "BiasMetric",
    "pii_leakage": "PIILeakageMetric",
}

#: Builds a metric from (name, threshold, judge, dataset); swapped out in tests.
MetricFactory = Callable[[str, float, Any, dict[str, Any]], Any]


def load_dataset(path: Path) -> dict[str, Any]:
    text = re.sub(
        r"\$\{(\w+)\}", lambda m: os.environ.get(m.group(1), m.group(0)), path.read_text()
    )
    dataset: dict[str, Any] = yaml.safe_load(text) or {}
    if not dataset.get("cases"):
        raise ValueError(f"{path} has no cases")
    dataset.setdefault("name", path.stem)
    return dataset


def deepeval_metric(name: str, threshold: float, judge: Any, dataset: dict[str, Any]) -> Any:
    import deepeval.metrics as metrics
    from deepeval.test_case import SingleTurnParams

    options = {"threshold": threshold, "model": judge, "async_mode": False}
    if name.startswith("geval:"):
        key = name.removeprefix("geval:")
        spec = (dataset.get("geval") or {}).get(key)
        if spec is None:
            raise ValueError(f"No geval definition named {key!r}")
        params = [SingleTurnParams(p) for p in spec.get("params", ["input", "actual_output"])]
        return metrics.GEval(
            name=key,
            criteria=spec.get("criteria"),
            evaluation_steps=spec.get("steps"),
            evaluation_params=params,
            **options,
        )
    if name not in METRICS:
        raise ValueError(f"Unknown metric {name!r}; known: {', '.join(sorted(METRICS))}, geval:*")
    return getattr(metrics, METRICS[name])(**options)


def test_case(case: dict[str, Any], seen: Observation) -> Any:
    from deepeval.test_case import LLMTestCase, ToolCall

    return LLMTestCase(
        input=case["input"],
        actual_output=seen.output,
        expected_output=case.get("expected_output"),
        context=case.get("context"),
        retrieval_context=seen.retrieval_context or case.get("context"),
        tools_called=[ToolCall(name=name) for name in seen.tools_called],
        expected_tools=[ToolCall(name=name) for name in case.get("expected_tools", [])],
        name=str(case.get("id", "")),
    )


def run(
    dataset: dict[str, Any],
    target: Target,
    judge: Any,
    *,
    metric_factory: MetricFactory = deepeval_metric,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Score every case; returns the summary and the per-case results."""
    defaults = dataset.get("defaults") or {}
    results: list[dict[str, Any]] = []
    for index, case in enumerate(dataset["cases"]):
        case_id = str(case.get("id", index))
        seen = target.run(case)
        record: dict[str, Any] = {
            "id": case_id,
            "input": case["input"],
            "output": seen.output,
            "tools_called": seen.tools_called,
            "latency_ms": round(seen.latency_ms, 1),
            "metrics": {},
            "error": seen.error,
        }
        if seen.error is None:
            threshold = float(case.get("threshold", defaults.get("threshold", 0.5)))
            built = test_case(case, seen)
            for name in case.get("metrics", defaults.get("metrics", [])):
                metric = metric_factory(name, threshold, judge, dataset)
                try:
                    metric.measure(built)
                    score = float(metric.score)
                    record["metrics"][name] = {
                        "score": round(score, 4),
                        "threshold": threshold,
                        "passed": bool(metric.is_successful()),
                        "reason": getattr(metric, "reason", None),
                    }
                except Exception as exc:  # a judge failure fails this metric, not the run
                    record["metrics"][name] = {
                        "score": 0.0,
                        "threshold": threshold,
                        "passed": False,
                        "reason": f"{type(exc).__name__}: {exc}",
                    }
        record["passed"] = seen.error is None and all(
            item["passed"] for item in record["metrics"].values()
        )
        results.append(record)
    return summarize(dataset["name"], results), results


def summarize(name: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    scores: dict[str, list[float]] = {}
    for record in results:
        for metric, item in record["metrics"].items():
            scores.setdefault(metric.replace(":", "_"), []).append(item["score"])
    passed = sum(1 for record in results if record["passed"])
    return {
        "dataset": name,
        "cases": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 4) if results else 0.0,
        "errors": sum(1 for record in results if record["error"]),
        "metrics": {name: round(statistics.fmean(values), 4) for name, values in scores.items()},
    }
