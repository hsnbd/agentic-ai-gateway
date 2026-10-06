"""The benchmark tooling: regression rules in bench_compare.py and retrieval scoring in rag_eval.py."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import bench_compare  # noqa: E402
import rag_eval  # noqa: E402


def test_flatten_drops_findings_level_and_notes() -> None:
    nested = {"cache": {"findings": {"hit_ratio": 0.5, "by": {"a": 1}}, "notes": ["x"]}}
    assert bench_compare.flatten(nested, "evaluate") == {
        "evaluate.cache.hit_ratio": 0.5,
        "evaluate.cache.by.a": 1,
    }


@pytest.mark.parametrize(
    ("rule", "baseline", "current", "fails"),
    [
        ({"min": 0.99}, None, 0.98, True),
        ({"min": 0.99}, None, 0.995, False),
        ({"max": 500}, None, 501, True),
        ({"equals": True}, True, False, True),
        ({"equals": False}, None, False, False),
        ({"max_drop": 0.05}, 0.90, 0.84, True),
        ({"max_drop": 0.05}, 0.90, 0.86, False),
        ({"max_increase": 1}, 10, 12, True),
        ({"max_increase_pct": 10}, 100, 111, True),
        ({"max_increase_pct": 10}, 100, 109, False),
        ({"max_drop_pct": 25}, 400, 299, True),
        ({"max_increase_pct": 10}, 0, 1, True),
        ({"max_increase_pct": 10}, 0, 0, False),
        ({"min": 1}, None, "not a number", False),
        ({"max_drop": 0.1}, "n/a", 0.5, False),
    ],
)
def test_check_rules(rule: dict, baseline: object, current: object, fails: bool) -> None:
    assert bool(bench_compare.check(rule, baseline, current)) is fails


def test_compare_flags_missing_gated_metrics_and_skips_optional_ones() -> None:
    rules = {
        "a": {"min": 1},
        "b": {"min": 1},
        "c": {"min": 1, "required": False},
    }
    rows, failed = bench_compare.compare({"b": 1}, {"a": 1}, rules)
    assert failed
    assert [row[0] for row in rows] == ["a", "b"]
    assert rows[1][3].startswith("FAIL: missing")


def _write_run(folder: Path, loadtest: dict) -> Path:
    folder.mkdir()
    (folder / "loadtest.json").write_text(json.dumps(loadtest))
    return folder


def test_main_passes_writes_step_summary_and_fails_on_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    thresholds = tmp_path / "t.yaml"
    thresholds.write_text("metrics:\n  loadtest.latency_p95_ms: {max_increase_pct: 20}\n")
    base = _write_run(tmp_path / "base", {"latency_p95_ms": 100})
    good = _write_run(tmp_path / "good", {"latency_p95_ms": 110})
    bad = _write_run(tmp_path / "bad", {"latency_p95_ms": 130})
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    assert bench_compare.main([str(base), str(good), "--thresholds", str(thresholds)]) == 0
    assert "No regressions" in capsys.readouterr().out
    assert "Benchmark comparison" in summary.read_text()

    monkeypatch.delenv("GITHUB_STEP_SUMMARY")
    assert bench_compare.main([str(base), str(bad), "--thresholds", str(thresholds)]) == 1
    assert "REGRESSION" in capsys.readouterr().out


def test_render_formats_missing_and_float_values() -> None:
    table = bench_compare.render([("m", None, 0.123456, "ok")])
    assert "| `m` | — | 0.1235 | ok |" in table


def test_rag_score_metrics() -> None:
    ranks = [1, 2, None, 6]
    metrics = rag_eval.score(ranks, 5)
    assert metrics["recall_at_1"] == 0.25
    assert metrics["recall_at_5"] == 0.5
    assert metrics["mrr"] == round((1 + 1 / 2 + 1 / 6) / 4, 4)
    assert metrics["ndcg_at_5"] == round((1 + 1 / math.log2(3)) / 4, 4)
    assert rag_eval.score([], 5)["mrr"] == 0.0


def test_rag_percentile() -> None:
    assert rag_eval._percentile([], 95) == 0.0
    assert rag_eval._percentile([5, 1, 3], 50) == 3
    assert rag_eval._percentile(list(range(1, 101)), 95) == 95


def test_bundled_dataset_is_consistent() -> None:
    dataset = json.loads(rag_eval.DEFAULT_DATASET.read_text())
    ids = {doc["id"] for doc in dataset["documents"]}
    assert len(ids) == len(dataset["documents"])
    assert {q["doc"] for q in dataset["questions"]} <= ids
    assert {q["kind"] for q in dataset["questions"]} == {"keyword", "paraphrase"}
