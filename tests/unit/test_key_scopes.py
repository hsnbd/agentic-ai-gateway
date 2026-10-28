"""Per-key collection allowlists, multi-collection search, and identity-bound tool arguments."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import ValidationError

from app.api.rag import RagQueryRequest, SearchRequest
from app.auth.keys import ResolvedKey, _deserialize_key, _serialize_key
from app.config.settings import Settings
from app.core.errors import InvalidRequestError, NotFoundError
from app.core.schemas import ChatRequest, Message, Role, ToolCall, ToolDef
from app.db.models import RagCollection
from app.db.session import Database
from app.rag.access import UNRESTRICTED, RagAccess
from app.rag.retrieve import RetrievedChunk
from app.rag.service import RagService
from app.tools.executor import ToolExecutor
from app.tools.governance import NO_BINDINGS, ToolBindings, ToolCallContext


@pytest.fixture
async def db() -> AsyncIterator[Database]:
    database = Database(Settings(database_url="sqlite+aiosqlite:///:memory:"))
    await database.startup()
    await database.create_all()
    yield database
    await database.shutdown()


def _key(**fields: Any) -> ResolvedKey:
    defaults: dict[str, Any] = {
        "id": "k1",
        "team_id": "t1",
        "is_active": True,
        "expires_at": None,
        "allowed_models": [],
        "blocked_models": [],
        "rpm_limit": None,
        "tpm_limit": None,
        "max_budget_usd": None,
        "spend_usd": 0,
        "guardrail_policy": None,
    }
    return ResolvedKey(**{**defaults, **fields})


# -- Collection allowlist ------------------------------------------------------


def test_key_snapshots_keep_the_new_scopes() -> None:
    key = _key(
        allowed_collections=["docs-*"],
        tool_arg_bindings={"crm__*": {"customer_id": "end_user"}},
    )
    restored = _deserialize_key(_serialize_key(key))
    assert restored.allowed_collections == ["docs-*"]
    assert restored.tool_arg_bindings == {"crm__*": {"customer_id": "end_user"}}


def test_allowlist_narrows_what_a_key_can_read() -> None:
    principal = SimpleNamespace(
        role="application",
        kind="virtual_key",
        team_id="t1",
        identifier="k1",
        key=_key(allowed_collections=["public-*", "c-exact"]),
    )
    access = RagAccess.for_principal(principal)  # type: ignore[arg-type]
    assert access.collections == ("public-*", "c-exact")
    public = RagCollection(id="c1", name="public-faq", owner_team_id="t1")
    by_id = RagCollection(id="c-exact", name="anything", owner_team_id="t1")
    private = RagCollection(id="c3", name="staff-notes", owner_team_id="t1")
    other_team = RagCollection(id="c4", name="public-other", owner_team_id="t2")
    assert access.can_read(public) and access.can_read(by_id)
    assert not access.can_read(private)
    assert not access.can_read(other_team)  # the allowlist never widens ownership
    assert UNRESTRICTED.can_read(private) and UNRESTRICTED.permits(private)

    ctx = SimpleNamespace(key_id="k1", team_id="t1", virtual_key=_key())
    assert RagAccess.for_context(ctx).collections == ()  # type: ignore[arg-type]


async def test_listing_applies_the_allowlist(db: Database) -> None:
    async with db.session() as session:
        for name in ("public-faq", "staff-notes"):
            session.add(RagCollection(name=name, embedding_model="e", embedding_dimensions=2))
    service = RagService(SimpleNamespace(db=db, settings=Settings()))  # type: ignore[arg-type]
    access = RagAccess(team_id="t1", key_id="k1", collections=("public-*",))
    assert [c.name for c in await service.list_collections(access)] == ["public-faq"]
    assert len(await service.list_collections()) == 2


# -- Multi-collection search -------------------------------------------------


def _search_service(collections: dict[str, RagCollection]) -> tuple[RagService, list[Any]]:
    service = RagService(
        SimpleNamespace(db=None, settings=Settings(rag_default_top_k=2))  # type: ignore[arg-type]
    )
    calls: list[Any] = []

    async def get_collection(collection_id: str, access: Any = UNRESTRICTED) -> RagCollection:
        if collection_id not in collections:
            raise NotFoundError(collection_id)
        return collections[collection_id]

    async def search(collection_id: str, query: str, **opts: Any) -> list[RetrievedChunk]:
        calls.append((collection_id, opts))
        base = {"a": 0.9, "b": 0.5}[collection_id]
        return [
            RetrievedChunk(id=f"{collection_id}{i}", text="t", score=base - i / 10)
            for i in range(2)
        ]

    service.get_collection = get_collection  # type: ignore[method-assign]
    service.search = search  # type: ignore[method-assign]
    return service, calls


def _collection(name: str, model: str = "e") -> RagCollection:
    return RagCollection(id=name, name=name, embedding_model=model, embedding_dimensions=2)


async def test_search_many_merges_by_score_and_tags_the_source() -> None:
    service, calls = _search_service({"a": _collection("a"), "b": _collection("b")})
    chunks = await service.search_many(["a", "b", "a"], "q", top_k=3, filters={"x": "y"})
    assert [c.id for c in chunks] == ["a0", "a1", "b0"]
    assert [c.metadata["collection_id"] for c in chunks] == ["a", "a", "b"]
    assert [call[0] for call in calls] == ["a", "b"]  # duplicates searched once
    assert calls[0][1]["filters"] == {"x": "y"}
    assert len(await service.search_many(["a", "b"], "q")) == 2  # default top_k


async def test_search_many_with_one_collection_is_a_plain_search() -> None:
    service, calls = _search_service({"a": _collection("a")})
    chunks = await service.search_many(["a"], "q", top_k=1)
    assert "collection_id" not in chunks[0].metadata
    assert calls == [("a", {"top_k": 1, "access": UNRESTRICTED})]


async def test_search_many_refuses_mixed_embedding_spaces_and_unknown_collections() -> None:
    service, _ = _search_service({"a": _collection("a"), "b": _collection("b", model="other")})
    with pytest.raises(InvalidRequestError, match="embedding model"):
        await service.search_many(["a", "b"], "q")
    with pytest.raises(NotFoundError):
        await service.search_many(["a", "missing"], "q")


def test_search_requests_name_one_or_more_collections() -> None:
    assert SearchRequest(collection_id="a", collection_ids=["b", "a"], query="q").targets() == [
        "a",
        "b",
    ]
    assert SearchRequest(collection_ids=["b"], query="q").targets() == ["b"]
    with pytest.raises(ValidationError, match="collection_id or collection_ids"):
        SearchRequest(query="q")
    request = ChatRequest(model="m", messages=[Message(role=Role.USER, content="x")])
    assert RagQueryRequest(collection_ids=["a", "b"], request=request).targets() == ["a", "b"]


# -- Tool argument bindings --------------------------------------------------


def _tool(name: str = "crm__get", required: list[str] | None = None) -> ToolDef:
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {"customer_id": {"type": "string"}, "query": {"type": "string"}},
    }
    if required is not None:
        parameters["required"] = required
    return ToolDef.model_validate(
        {
            "type": "function",
            "function": {"name": name, "description": "d", "parameters": parameters},
        }
    )


def test_bindings_match_tools_and_declared_arguments() -> None:
    bindings = ToolBindings.for_key(
        SimpleNamespace(
            tool_arg_bindings={
                "crm__*": {"customer_id": "end_user", "missing": "key_id"},
                "crm__get": {"customer_id": "team_id"},
                "broken": "not-a-mapping",
            }
        )
    )
    assert bindings.for_tool("crm__get") == {"customer_id": "team_id", "missing": "key_id"}
    assert bindings.for_tool("crm__list") == {"customer_id": "end_user", "missing": "key_id"}
    assert bindings.for_tool("billing__get") == {}
    assert ToolBindings.for_key(None) == NO_BINDINGS


def test_bound_arguments_are_hidden_from_the_model() -> None:
    bindings = ToolBindings(rules=(("crm__*", (("customer_id", "end_user"),)),))
    hidden = bindings.hide(_tool(required=["customer_id", "query"]))
    assert list(hidden.function.parameters["properties"]) == ["query"]
    assert hidden.function.parameters["required"] == ["query"]
    assert hidden.function.description == "d"
    assert "required" not in bindings.hide(_tool()).function.parameters
    unbound = _tool("other__get")
    assert bindings.hide(unbound) is unbound


class _Registry:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def tools_for(self, server_ids: Any = None) -> list[ToolDef]:
        return [_tool(required=["customer_id"])]

    def resolve(self, name: str) -> tuple[str, str]:
        return "crm", "get"

    def server_name(self, server_id: str) -> str:
        return server_id

    async def call_tool(self, server_id: str, name: str, arguments: Any) -> dict[str, Any]:
        self.calls.append(arguments)
        return {"content": [{"type": "text", "text": json.dumps(arguments)}]}


class _Db:
    def __init__(self) -> None:
        self.rows: list[Any] = []

    def session(self) -> Any:
        rows = self.rows

        class _Session:
            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                return None

            def add(self, row: Any) -> None:
                rows.append(row)

        return _Session()


def _executor(
    rules: dict[str, dict[str, str]], end_user: str | None = "cust-7"
) -> tuple[ToolExecutor, _Registry, _Db]:
    registry, database = _Registry(), _Db()
    state = SimpleNamespace(db=database, components={}, settings=None)
    context = ToolCallContext(
        state=state,
        source="direct",
        key_id="k1",
        team_id="t1",
        end_user=end_user,
        bindings=ToolBindings.for_key(SimpleNamespace(tool_arg_bindings=rules)),
    )
    executor = ToolExecutor(registry, call_context=context)  # type: ignore[arg-type]
    return executor, registry, database


async def test_bound_arguments_overwrite_what_the_model_sent() -> None:
    executor, registry, database = _executor({"crm__*": {"customer_id": "end_user"}})
    message = await executor.execute(
        ToolCall(name="crm__get", arguments='{"customer_id": "someone-else", "query": "x"}')
    )
    assert registry.calls == [{"customer_id": "cust-7", "query": "x"}]
    assert json.loads(message.text())["customer_id"] == "cust-7"
    assert database.rows[0].bound_args == ["customer_id"]
    assert database.rows[0].status == "ok"


async def test_identity_sources_and_removed_arguments() -> None:
    executor, registry, _ = _executor({"crm__*": {"customer_id": "team_id"}})
    await executor.execute(ToolCall(name="crm__get", arguments="{}"))
    assert registry.calls == [{"customer_id": "t1"}]

    executor, registry, database = _executor({"crm__*": {"customer_id": "none"}})
    message = await executor.execute(ToolCall(name="crm__get", arguments='{"customer_id": "c-1"}'))
    # Removed, so the required argument is now missing and validation refuses it.
    assert registry.calls == [] and "customer_id is required" in message.text()
    assert database.rows[0].status == "invalid" and database.rows[0].bound_args == []


async def test_a_binding_without_an_identity_refuses_the_call() -> None:
    executor, registry, database = _executor({"crm__*": {"customer_id": "end_user"}}, end_user=None)
    message = await executor.execute(ToolCall(name="crm__get", arguments="{}"))
    assert "needs the caller's end_user" in message.text()
    assert registry.calls == [] and database.rows[0].status == "unbound"


async def test_undeclared_bound_arguments_are_stripped_not_filled() -> None:
    """A server may accept arguments its schema omits; the model still cannot set them."""
    executor, registry, database = _executor(
        {"crm__*": {"customer_id": "end_user", "tenant": "key_id"}}, end_user="cust-7"
    )
    await executor.execute(
        ToolCall(name="crm__get", arguments='{"customer_id": "x", "tenant": "evil", "query": "q"}')
    )
    assert registry.calls == [{"customer_id": "cust-7", "query": "q"}]
    assert database.rows[0].bound_args == ["customer_id"]
    # Without an end user, only declared bindings need one.
    executor, registry, _ = _executor({"crm__*": {"tenant": "end_user"}}, end_user=None)
    await executor.execute(
        ToolCall(name="crm__get", arguments='{"customer_id": "c", "tenant": "t"}')
    )
    assert registry.calls == [{"customer_id": "c"}]


async def test_unparseable_arguments_are_left_to_validation() -> None:
    for raw in ("{not json", "[1, 2]"):
        executor, registry, database = _executor({"crm__*": {"customer_id": "end_user"}})
        await executor.execute(ToolCall(name="crm__get", arguments=raw))
        assert registry.calls == [] and database.rows[0].status == "invalid"


async def test_executors_without_a_context_bind_nothing() -> None:
    registry = _Registry()
    executor = ToolExecutor(registry)  # type: ignore[arg-type]
    await executor.execute(ToolCall(name="crm__get", arguments='{"customer_id": "c"}'))
    assert registry.calls == [{"customer_id": "c"}]


def test_binding_values_come_from_the_call_context() -> None:
    context = ToolCallContext(state=None, source="agent", key_id="k", team_id="t", end_user="u")
    assert [context.binding_value(s) for s in ("end_user", "key_id", "team_id", "none")] == [
        "u",
        "k",
        "t",
        None,
    ]


async def test_chat_grounding_searches_extra_collections() -> None:
    from app.core.schemas import RagOptions
    from app.rag.stage import retrieve_and_augment

    class _Service:
        async def search_many(self, ids: list[str], query: str, **opts: Any) -> Any:
            assert ids == ["a", "b"] and query == "question"
            return [RetrievedChunk(id="x", text="grounding", score=1.0)]

    request = ChatRequest(model="m", messages=[Message(role=Role.USER, content="question")])
    augmented, sources = await retrieve_and_augment(
        _Service(),  # type: ignore[arg-type]
        request,
        RagOptions(collection_id="a", extra_collection_ids=["b"]),
    )
    assert sources[0]["id"] == "x" and "grounding" in augmented.messages[0].text()
