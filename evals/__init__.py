"""Answer-quality evaluation with DeepEval, judged through the gateway itself.

A dataset (YAML) names a *target* to ask (the gateway's chat API, optionally
with RAG grounding or the MCP agent loop, or any HTTP application built on
the gateway), the judge model, and test cases with the DeepEval metrics to
score them by. The runner writes a summary that ``scripts/bench_compare.py``
gates against ``bench/thresholds.yaml``::

    uv run python -m evals bench/datasets/eval_gateway.yaml --out bench/results/eval

See docs/evaluation.md ("Answer quality").
"""
