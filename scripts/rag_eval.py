"""Retrieval-quality benchmark for the gateway's RAG collections.

Ingests a labelled corpus into a throwaway collection, asks every question
through ``POST /v1/rag/search``, and scores where the expected document ranks:

- recall@1 and recall@k: the expected document is in the top 1 / top k results
- MRR: mean reciprocal rank of the expected document
- nDCG@k: rank-discounted gain, with one relevant document per question
- retrieval latency p50 / p95

Each retrieval ``mode`` (``vector``, ``hybrid``) is scored separately, and
broken down by question ``kind`` so the effect of lexical search on
code-and-name questions is visible. Like the other benchmark scripts it only
talks HTTP, so the numbers describe the deployed system::

    uv run python scripts/rag_eval.py --base-url http://localhost:18000 --api-key sk-...

Results depend on the collection's embedding model. Against the e2e stack's
hash-based fake embeddings they measure the retrieval pipeline, not semantic
quality; point it at a gateway with a real embedding model for that.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import httpx

DEFAULT_DATASET = Path(__file__).resolve().parent.parent / "bench" / "datasets" / "rag_eval.json"


def score(ranks: list[int | None], k: int) -> dict[str, float]:
    """Aggregate 1-based ranks (None = not retrieved) into retrieval metrics."""
    total = len(ranks) or 1
    found = [rank for rank in ranks if rank is not None]
    return {
        "recall_at_1": round(sum(1 for rank in found if rank == 1) / total, 4),
        f"recall_at_{k}": round(sum(1 for rank in found if rank <= k) / total, 4),
        "mrr": round(sum(1 / rank for rank in found) / total, 4),
        f"ndcg_at_{k}": round(
            sum(1 / math.log2(rank + 1) for rank in found if rank <= k) / total, 4
        ),
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(pct / 100 * len(ordered)) - 1))]


class Evaluator:
    def __init__(self, client: httpx.Client, dataset: dict[str, Any], k: int) -> None:
        self.client = client
        self.dataset = dataset
        self.k = k
        self.doc_ids: dict[str, str] = {}
        self.collection_id = ""

    def setup(self, embedding_model: str | None) -> None:
        body: dict[str, Any] = {
            "name": f"rag-eval-{uuid.uuid4().hex[:8]}",
            "description": "rag_eval.py benchmark",
        }
        if embedding_model:
            body["embedding_model"] = embedding_model
        created = self.client.post("/v1/rag/collections", json=body)
        created.raise_for_status()
        self.collection_id = created.json()["id"]
        for doc in self.dataset["documents"]:
            res = self.client.post(
                f"/v1/rag/collections/{self.collection_id}/documents",
                json={
                    "title": doc["title"],
                    "content": doc["content"],
                    "source": f"{doc['id']}.md",
                },
            )
            res.raise_for_status()
            self.doc_ids[res.json()["id"]] = doc["id"]

    def teardown(self) -> None:
        if self.collection_id:
            self.client.delete(f"/v1/rag/collections/{self.collection_id}")

    def run_mode(self, mode: str) -> dict[str, Any]:
        ranks: list[int | None] = []
        by_kind: dict[str, list[int | None]] = {}
        latencies: list[float] = []
        for question in self.dataset["questions"]:
            body: dict[str, Any] = {
                "collection_id": self.collection_id,
                "query": question["query"],
                "top_k": self.k,
            }
            if mode != "vector":
                body["search_mode"] = mode
            started = time.perf_counter()
            res = self.client.post("/v1/rag/search", json=body)
            latencies.append((time.perf_counter() - started) * 1000)
            if res.status_code == 422 and mode != "vector":
                return {"supported": False, "detail": res.text[:300]}
            res.raise_for_status()
            ordered: list[str] = []
            for result in res.json()["results"]:
                doc = self.doc_ids.get(result.get("document_id") or "")
                if doc and doc not in ordered:
                    ordered.append(doc)
            rank = ordered.index(question["doc"]) + 1 if question["doc"] in ordered else None
            ranks.append(rank)
            by_kind.setdefault(question.get("kind", "other"), []).append(rank)
        return {
            **score(ranks, self.k),
            "latency_p50_ms": round(statistics.median(latencies), 2),
            "latency_p95_ms": round(_percentile(latencies, 95), 2),
            "by_kind": {kind: score(values, self.k) for kind, values in sorted(by_kind.items())},
        }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--base-url", default="http://localhost:18000")
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--modes", default="vector,hybrid", help="Comma-separated retrieval modes")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-model", help="Collection embedding model (default: gateway's)")
    parser.add_argument(
        "--min-recall-at-k", type=float, default=0.0, help="Fail below this for any mode"
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    dataset = json.loads(args.dataset.read_text())
    headers = {"Authorization": f"Bearer {args.api_key}"}
    report: dict[str, Any] = {
        "dataset": args.dataset.name,
        "documents": len(dataset["documents"]),
        "questions": len(dataset["questions"]),
        "top_k": args.top_k,
    }
    with httpx.Client(
        base_url=args.base_url.rstrip("/"), headers=headers, timeout=args.timeout
    ) as client:
        evaluator = Evaluator(client, dataset, args.top_k)
        try:
            evaluator.setup(args.embedding_model)
            for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
                report[mode] = evaluator.run_mode(mode)
        finally:
            evaluator.teardown()

    recall_key = f"recall_at_{args.top_k}"
    failures = [
        f"{mode} {recall_key} {result[recall_key]} < {args.min_recall_at_k}"
        for mode, result in report.items()
        if isinstance(result, dict)
        and recall_key in result
        and result[recall_key] < args.min_recall_at_k
    ]
    report["passed"] = not failures

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            f"RAG retrieval benchmark: {report['documents']} documents, "
            f"{report['questions']} questions, k={args.top_k}"
        )
        for mode, result in report.items():
            if not isinstance(result, dict) or mode == "by_kind":
                continue
            if not result.get("supported", True):
                print(f"\n{mode}: not supported by this gateway")
                continue
            print(f"\n{mode}")
            for name, value in result.items():
                if name != "by_kind":
                    print(f"  {name:<18} {value}")
            for kind, values in result["by_kind"].items():
                print(f"  [{kind}] recall@{args.top_k} {values[recall_key]}  mrr {values['mrr']}")
        for failure in failures:
            print(f"\nFAIL: {failure}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
