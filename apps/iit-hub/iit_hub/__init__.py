"""IIT information hub: a LangGraph agent built on the AI gateway.

The hub owns the conversation (memory, who the user is, approvals); the
gateway owns everything model- and data-facing: routing and fallback,
caching, guardrails, budgets, RAG over IIT's documents, and governed calls to
IIT's student information system. Each user role has its own gateway key, so
the gateway enforces what each role may read and do.
"""
