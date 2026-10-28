"""The DeepEval harness (evals/): judge parsing, targets, dataset loading, and the runner.

HTTP is served by httpx MockTransport and metrics are fakes, so nothing here
calls a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from evals import __main__ as cli
from evals.judge import JUDGE_TAG, GatewayJudge, parse_reply
from evals.runner import deepeval_metric, load_dataset, run, summarize
from evals.targets import GatewayChatTarget, HttpTarget, Observation, build_target, fill, select


class Verdict(BaseModel):
    score: float


def test_parse_reply_extracts_json_from_prose_and_fences() -> None:
    assert parse_reply("plain", None) == "plain"
    assert parse_reply('```json\n{"score": 1}\n```', Verdict) == Verdict(score=1)
    assert parse_reply('Sure! {"score": 0.5} hope that helps', Verdict) == Verdict(score=0.5)
    with pytest.raises(ValueError):
        parse_reply("no json here", Verdict)


def _completion(text: str) -> Any:
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


async def test_gateway_judge_tags_its_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    judge = GatewayJudge("judge-model", base_url="http://gw/", api_key="k")
    assert judge.get_model_name() == "gateway:judge-model" and judge.load_model() is judge
    seen: list[dict[str, Any]] = []

    class _Completions:
        def create(self, **kwargs: Any) -> Any:
            seen.append(kwargs)
            return _completion('{"score": 1}')

    class _AsyncCompletions:
        async def create(self, **kwargs: Any) -> Any:
            seen.append(kwargs)
            return _completion("text")

    judge._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))  # type: ignore[assignment]
    assert judge.generate("p", Verdict) == Verdict(score=1)
    assert seen[0]["extra_body"] == {"aigw": {"tags": [JUDGE_TAG]}}
    assert seen[0]["model"] == "judge-model" and seen[0]["temperature"] == 0

    created: list[dict[str, Any]] = []

    def async_client(**kwargs: Any) -> Any:
        created.append(kwargs)
        return SimpleNamespace(chat=SimpleNamespace(completions=_AsyncCompletions()))

    monkeypatch.setattr("evals.judge.openai.AsyncOpenAI", async_client)
    assert await judge.a_generate("p") == "text"
    assert await judge.a_generate("p") == "text"  # the async client is created once
    assert created == [{"base_url": "http://gw/v1", "api_key": "k", "max_retries": 2}]


def test_select_and_fill() -> None:
    data = {"a": {"b": [{"c": 1}, {"c": 2}]}, "list": [5]}
    assert select(data, "a.b[*].c") == [1, 2]
    assert select(data, "a.b[1].c") == [2]
    assert select(data, "a.b[9].c") == [] and select(data, "missing.x") == []
    assert select(data, "list[0]") == [5] and select(data, None) == []
    assert select({"a": 1}, "a[0]") == []
    values = {"input": "hi", "id": 7}
    assert fill(
        {"q": "{input}", "n": "{id}", "s": "t-{id}-{other}", "l": ["{input}", 3]}, values
    ) == {
        "q": "hi",
        "n": 7,
        "s": "t-7-{other}",
        "l": ["hi", 3],
    }


def _client(handler: Any) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_gateway_target_reads_sources_and_executed_tools() -> None:
    bodies: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        assert request.headers["authorization"] == "Bearer key"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": "answer"}}],
                "aigw": {"sources": [{"text": "ctx"}], "tool_executions": [{"name": "t__x"}]},
            },
        )

    target = GatewayChatTarget(
        base_url="http://gw/",
        api_key="key",
        model="m",
        system="Be brief.",
        aigw={"mcp": {}},
        client=_client(handler),
    )
    seen = target.run({"input": "q", "user": "u1", "context": ["fact"], "aigw": {"tags": ["x"]}})
    assert (seen.output, seen.retrieval_context, seen.tools_called) == ("answer", ["ctx"], ["t__x"])
    body = bodies[0]
    assert body["user"] == "u1" and body["aigw"] == {"no_cache": True, "mcp": {}, "tags": ["x"]}
    assert body["messages"][0]["content"].startswith("Be brief.\n\nAnswer using only this context")

    target.run({"input": "q", "context": ["fact"], "aigw": {"rag": {"collection_id": "c"}}})
    assert bodies[1]["messages"][0] == {"role": "system", "content": "Be brief."}
    bare = GatewayChatTarget(
        base_url="http://gw", api_key="key", model="m", client=_client(handler)
    )
    bare.run({"input": "q"})
    assert bodies[2]["messages"] == [{"role": "user", "content": "q"}] and "user" not in bodies[2]


def test_gateway_target_reports_http_errors() -> None:
    target = GatewayChatTarget(
        base_url="http://gw",
        api_key="k",
        model="m",
        client=_client(lambda request: httpx.Response(429, text="slow down")),
    )
    assert target.run({"input": "q"}).error == "HTTP 429: slow down"


def test_http_target_opens_a_thread_then_asks() -> None:
    calls: list[tuple[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.url.path, json.loads(request.content) if request.content else None))
        if request.url.path == "/threads":
            return httpx.Response(200, json={"thread_id": "t1"})
        return httpx.Response(
            200,
            json={"answer": "hello", "sources": [{"text": "s"}], "tools": ["enroll"]},
        )

    target = HttpTarget(
        base_url="http://hub/",
        start={"path": "/threads", "body": {"role": "guest"}, "id_path": "thread_id"},
        request={"path": "/threads/{id}/messages", "body": {"content": "{input}", "x": "{v}"}},
        output_path="answer",
        context_path="sources[*].text",
        tools_path="tools[*]",
        headers={"x-key": "1"},
        client=_client(handler),
    )
    seen = target.run({"input": "hi", "vars": {"v": 3}, "start_body": {"role": "teacher"}})
    assert (seen.output, seen.retrieval_context, seen.tools_called) == ("hello", ["s"], ["enroll"])
    assert calls == [
        ("/threads", {"role": "teacher"}),
        ("/threads/t1/messages", {"content": "hi", "x": 3}),
    ]

    plain = HttpTarget(
        base_url="http://hub",
        request={"method": "POST", "path": "/ask", "body": {"q": "{input}"}},
        output_path="nothing",
        client=_client(handler),
    )
    assert plain.run({"input": "hi"}).output == ""
    overridden = plain.run({"input": "hi", "output_path": "answer", "tools_path": "tools[*]"})
    assert (overridden.output, overridden.tools_called) == ("hello", ["enroll"])

    failing = HttpTarget(
        base_url="http://hub",
        request={"path": "/ask"},
        output_path="answer",
        client=_client(lambda request: httpx.Response(500)),
    )
    assert "HTTPStatusError" in (failing.run({"input": "hi"}).error or "")


def test_build_target() -> None:
    gateway = build_target({"model": "x"}, base_url="http://gw", api_key="k", model="default")
    assert isinstance(gateway, GatewayChatTarget) and gateway.model == "x"
    http = build_target(
        {"type": "http", "base_url": "http://hub", "request": {"path": "/a"}, "output_path": "o"},
        base_url="",
        api_key="",
        model="",
    )
    assert isinstance(http, HttpTarget)
    with pytest.raises(ValueError, match="Unknown target type"):
        build_target({"type": "grpc"}, base_url="", api_key="", model="")


def test_load_dataset_expands_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EVAL_TEST_MODEL", "m-1")
    path = tmp_path / "set.yaml"
    path.write_text(
        'target:\n  model: ${EVAL_TEST_MODEL}\n  other: "${UNSET_VAR_XYZ}"\ncases: [{input: q}]'
    )
    dataset = load_dataset(path)
    assert dataset["name"] == "set" and dataset["target"]["model"] == "m-1"
    assert dataset["target"]["other"] == "${UNSET_VAR_XYZ}"
    (tmp_path / "empty.yaml").write_text("name: x\n")
    with pytest.raises(ValueError, match="no cases"):
        load_dataset(tmp_path / "empty.yaml")


class _Target:
    def __init__(self, observations: list[Observation]) -> None:
        self.observations = observations

    def run(self, case: dict[str, Any]) -> Observation:
        return self.observations.pop(0)


class _Metric:
    def __init__(self, name: str, threshold: float, scores: dict[str, float]) -> None:
        self.name, self.threshold, self.scores = name, threshold, scores
        self.reason = f"because {name}"

    def measure(self, test_case: Any) -> None:
        if self.name == "boom":
            raise RuntimeError("judge down")
        self.score = self.scores[self.name]

    def is_successful(self) -> bool:
        return bool(self.score >= self.threshold)


def test_run_scores_cases_and_summarises() -> None:
    dataset = {
        "name": "d",
        "defaults": {"metrics": ["good"], "threshold": 0.5},
        "cases": [
            {"id": "a", "input": "q1", "expected_tools": ["t"]},
            {"id": "b", "input": "q2", "metrics": ["good", "bad", "geval:x"], "threshold": 0.6},
            {"input": "q3", "metrics": ["boom"]},
            {"id": "d", "input": "q4"},
        ],
    }
    scores = {"good": 0.9, "bad": 0.1, "geval:x": 1.0}
    built: list[Any] = []

    def factory(name: str, threshold: float, judge: Any, data: dict[str, Any]) -> _Metric:
        assert judge == "judge" and data is dataset
        return _Metric(name, threshold, scores)

    target = _Target(
        [
            Observation(output="o1", tools_called=["t"], retrieval_context=["c"]),
            Observation(output="o2"),
            Observation(output="o3"),
            Observation(error="HTTP 500"),
        ]
    )
    import evals.runner as runner

    original = runner.test_case
    runner.test_case = lambda case, seen: built.append((case, seen)) or original(case, seen)  # type: ignore[assignment,func-returns-value]
    try:
        summary, results = run(dataset, target, "judge", metric_factory=factory)
    finally:
        runner.test_case = original  # type: ignore[assignment]
    assert [r["passed"] for r in results] == [True, False, False, False]
    assert results[1]["metrics"]["bad"] == {
        "score": 0.1,
        "threshold": 0.6,
        "passed": False,
        "reason": "because bad",
    }
    assert results[2]["id"] == "2" and "judge down" in results[2]["metrics"]["boom"]["reason"]
    assert results[3]["metrics"] == {} and results[3]["error"] == "HTTP 500"
    assert summary == {
        "dataset": "d",
        "cases": 4,
        "passed": 1,
        "pass_rate": 0.25,
        "errors": 1,
        "metrics": {"good": 0.9, "bad": 0.1, "geval_x": 1.0, "boom": 0.0},
    }
    case_a = built[0][1]
    assert case_a.tools_called == ["t"]
    assert summarize("empty", [])["pass_rate"] == 0.0


def test_deepeval_metrics_are_built_from_names() -> None:
    judge = GatewayJudge("j", base_url="http://gw", api_key="k")
    dataset = {
        "geval": {"polite": {"criteria": "Be polite.", "params": ["input", "actual_output"]}}
    }
    relevancy = deepeval_metric("answer_relevancy", 0.7, judge, dataset)
    assert type(relevancy).__name__ == "AnswerRelevancyMetric" and relevancy.threshold == 0.7
    geval = deepeval_metric("geval:polite", 0.5, judge, dataset)
    assert type(geval).__name__ == "GEval"
    with pytest.raises(ValueError, match="No geval definition"):
        deepeval_metric("geval:missing", 0.5, judge, {})
    with pytest.raises(ValueError, match="Unknown metric"):
        deepeval_metric("vibes", 0.5, judge, {})


def test_test_cases_carry_tools_and_context() -> None:
    from evals.runner import test_case

    built = test_case(
        {"id": 1, "input": "q", "context": ["c"], "expected_tools": ["a"]},
        Observation(output="o", tools_called=["a", "b"]),
    )
    assert [t.name for t in built.tools_called] == ["a", "b"]
    assert [t.name for t in built.expected_tools] == ["a"]
    assert built.retrieval_context == ["c"] and built.name == "1"


def test_cli_writes_summary_and_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "set.yaml"
    path.write_text("target: {model: m}\njudge: {}\ncases: [{input: q}]")
    seen: dict[str, Any] = {}

    def fake_run(dataset: dict[str, Any], target: Any, judge: Any) -> Any:
        seen.update(target=target, judge=judge)
        record = {"id": "0", "passed": True, "metrics": {"m": {"score": 1.0}}, "error": None}
        return {"passed": 1, "cases": 1}, [record]

    monkeypatch.setattr(cli, "run", fake_run)
    out = tmp_path / "out"
    assert cli.main([str(path), "--out", str(out), "--model", "override", "--api-key", "k"]) == 0
    assert json.loads((out / "eval.json").read_text()) == {"passed": 1, "cases": 1}
    assert seen["target"].model == "override" and seen["judge"].name == "override"

    monkeypatch.setattr(
        cli,
        "run",
        lambda *a: (
            {"passed": 0, "cases": 1},
            [{"id": "0", "passed": False, "metrics": {}, "error": "HTTP 500"}],
        ),
    )
    assert cli.main([str(path), "--out", str(out), "--judge-model", "j", "--api-key", "k"]) == 1
    with pytest.raises(SystemExit):
        cli.main([str(path), "--out", str(out), "--api-key", ""])

    (tmp_path / "nomodel.yaml").write_text("cases: [{input: q}]")
    with pytest.raises(SystemExit):
        cli.main([str(tmp_path / "nomodel.yaml"), "--out", str(out), "--api-key", "k"])
