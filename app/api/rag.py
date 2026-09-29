"""HTTP API for gateway-owned retrieval-augmented generation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from starlette.datastructures import UploadFile

from app.api.deps import CurrentPrincipal, require_gateway_writer
from app.core.errors import InvalidRequestError, NotFoundError
from app.core.pipeline import RequestContext
from app.core.schemas import ChatRequest, ChatResponse
from app.core.state import GatewayState
from app.db.models import RagChunk, RagDocument
from app.rag.retrieve import RetrievedChunk, augment_request, build_context
from app.rag.service import DEFAULT_CHUNK_PAGE_SIZE, MAX_CHUNK_PAGE_SIZE, RagService

router = APIRouter()

#: Mutating and billable routes: console viewers are read-only.
_WRITE = [Depends(require_gateway_writer)]

#: Number of leading floats returned in an embedding preview.
EMBEDDING_PREVIEW_VALUES = 8


class CollectionCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    description: str | None = None
    embedding_model: str | None = None
    embedding_dimensions: int | None = Field(default=None, gt=0)
    chunk_size: int | None = Field(default=None, gt=0)
    chunk_overlap: int | None = Field(default=None, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CollectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    name: str
    description: str | None
    embedding_model: str
    embedding_dimensions: int
    chunk_size: int
    chunk_overlap: int
    document_count: int
    chunk_count: int
    metadata: dict[str, Any] = Field(validation_alias="metadata_")
    created_at: datetime
    updated_at: datetime


class DocumentIngestRequest(BaseModel):
    content: str
    source: str = ""
    title: str | None = None
    content_type: str = "text/plain"
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    collection_id: str
    title: str
    source: str | None
    content_type: str
    content_hash: str
    byte_size: int
    status: str
    #: Populated when ingestion failed, so the console can show the reason.
    error_message: str | None = None
    chunk_count: int
    metadata: dict[str, Any] = Field(validation_alias="metadata_")
    created_at: datetime
    updated_at: datetime
    #: When ingestion finished; null while the document is pending or failed.
    ingested_at: datetime | None = None

    @classmethod
    def of(cls, document: RagDocument) -> DocumentResponse:
        response = cls.model_validate(document)
        if document.status == "ready":
            response.ingested_at = document.updated_at
        return response


class EmbeddingPreview(BaseModel):
    """Vector shape plus a short prefix; full vectors are never returned."""

    dimensions: int
    preview: list[float]
    truncated: bool


class ChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True, populate_by_name=True)

    id: str
    collection_id: str
    document_id: str
    chunk_index: int
    text: str = Field(validation_alias="content")
    token_count: int
    char_count: int = 0
    vector_id: str | None = None
    embedding: EmbeddingPreview | None = None
    metadata: dict[str, Any] = Field(validation_alias="metadata_")
    created_at: datetime

    @classmethod
    def of(cls, chunk: RagChunk, vector: list[float] | None = None) -> ChunkResponse:
        response = cls.model_validate(chunk)
        response.char_count = len(chunk.content)
        if vector is not None:
            response.embedding = EmbeddingPreview(
                dimensions=len(vector),
                preview=[round(value, 6) for value in vector[:EMBEDDING_PREVIEW_VALUES]],
                truncated=len(vector) > EMBEDDING_PREVIEW_VALUES,
            )
        return response


class ChunkListResponse(BaseModel):
    collection_id: str
    document_id: str | None = None
    total: int
    limit: int
    offset: int
    items: list[ChunkResponse]


class SearchRequest(BaseModel):
    collection_id: str
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, gt=0, le=100)
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: dict[str, str] | None = None


class RetrievedChunkResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    text: str
    score: float
    source: str | None = None
    document_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    collection_id: str
    results: list[RetrievedChunkResponse]


class RagQueryRequest(BaseModel):
    collection_id: str
    request: ChatRequest
    query: str | None = None
    top_k: int = Field(default=5, gt=0, le=100)
    min_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    diversity: float = Field(default=0.0, ge=0.0, le=1.0)
    filters: dict[str, str] | None = None
    max_context_tokens: int = Field(default=4000, gt=0)
    mode: str = "system"


class RagQueryResponse(BaseModel):
    response: ChatResponse
    sources: list[RetrievedChunkResponse]


class DeleteResponse(BaseModel):
    deleted: bool


class DeleteDocumentsResponse(BaseModel):
    deleted: int


def _state(request: Request) -> GatewayState:
    state: GatewayState = request.app.state.gateway
    return state


def _service(request: Request) -> RagService:
    state = _state(request)
    service = state.components.get("rag_service")
    if service is None:
        service = RagService(state)
        state.components["rag_service"] = service
    return service


def _chunk_response(chunks: list[RetrievedChunk]) -> list[RetrievedChunkResponse]:
    return [RetrievedChunkResponse.model_validate(chunk) for chunk in chunks]


@router.post(
    "/v1/rag/collections",
    dependencies=_WRITE,
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_collection(payload: CollectionCreate, request: Request) -> CollectionResponse:
    service = _service(request)
    collection = await service.create_collection(
        payload.name,
        description=payload.description,
        embedding_model=payload.embedding_model,
        embedding_dimensions=payload.embedding_dimensions,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        metadata=payload.metadata,
    )
    return CollectionResponse.model_validate(collection)


@router.get("/v1/rag/collections", response_model=list[CollectionResponse])
async def list_collections(request: Request) -> list[CollectionResponse]:
    return [
        CollectionResponse.model_validate(item)
        for item in await _service(request).list_collections()
    ]


@router.get("/v1/rag/collections/{collection_id}", response_model=CollectionResponse)
async def get_collection(collection_id: str, request: Request) -> CollectionResponse:
    collection = await _service(request).get_collection(collection_id)
    return CollectionResponse.model_validate(collection)


@router.delete(
    "/v1/rag/collections/{collection_id}", dependencies=_WRITE, response_model=DeleteResponse
)
async def delete_collection(collection_id: str, request: Request) -> DeleteResponse:
    await _service(request).delete_collection(collection_id)
    return DeleteResponse(deleted=True)


_DOCUMENT_REQUEST_BODY = {
    "requestBody": {
        "required": True,
        "content": {
            "application/json": {
                "schema": {
                    "type": "object",
                    "required": ["content"],
                    "properties": {
                        "content": {"type": "string"},
                        "source": {"type": "string", "default": ""},
                        "title": {"type": "string", "nullable": True},
                        "content_type": {"type": "string", "default": "text/plain"},
                        "metadata": {"type": "object", "additionalProperties": True},
                    },
                }
            },
            "multipart/form-data": {
                "schema": {
                    "type": "object",
                    "required": ["file"],
                    "properties": {
                        "file": {"type": "string", "format": "binary"},
                        "metadata": {"type": "string", "description": "Optional JSON object"},
                    },
                }
            },
        },
    }
}


@router.post(
    "/v1/rag/collections/{collection_id}/documents",
    dependencies=_WRITE,
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    openapi_extra=_DOCUMENT_REQUEST_BODY,
)
async def ingest_document_route(collection_id: str, request: Request) -> DocumentResponse:
    service = _service(request)
    content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0].lower()
    if content_type == "application/json":
        payload = DocumentIngestRequest.model_validate(await request.json())
        content = payload.content
        source = payload.source
        metadata = {**payload.metadata, "content_type": payload.content_type}
        if payload.title:
            metadata["title"] = payload.title
    elif content_type == "multipart/form-data":
        form = await request.form()
        uploaded = form.get("file")
        if not isinstance(uploaded, UploadFile):
            raise InvalidRequestError("Multipart document ingestion requires a file field")
        filename = uploaded.filename or ""
        if not filename.lower().endswith((".txt", ".md")):
            raise InvalidRequestError("Only .txt and .md files can be ingested")
        content = (await uploaded.read()).decode("utf-8", errors="replace")
        source = filename
        media_type = "text/markdown" if filename.lower().endswith(".md") else "text/plain"
        metadata = {"content_type": media_type}
        raw_metadata = form.get("metadata")
        if raw_metadata:
            import json

            try:
                extra = json.loads(str(raw_metadata))
            except json.JSONDecodeError as exc:
                raise InvalidRequestError("File metadata must be valid JSON") from exc
            if not isinstance(extra, dict):
                raise InvalidRequestError("File metadata must be a JSON object")
            metadata.update(extra)
    else:
        raise InvalidRequestError("Use application/json or multipart/form-data")

    document = await service.ingest(collection_id, content, source=source, metadata=metadata)
    return DocumentResponse.of(document)


@router.get(
    "/v1/rag/collections/{collection_id}/documents",
    response_model=list[DocumentResponse],
)
async def list_documents(collection_id: str, request: Request) -> list[DocumentResponse]:
    return [
        DocumentResponse.of(item) for item in await _service(request).list_documents(collection_id)
    ]


@router.delete(
    "/v1/rag/collections/{collection_id}/documents",
    dependencies=_WRITE,
    response_model=DeleteDocumentsResponse,
)
async def delete_documents(collection_id: str, request: Request) -> DeleteDocumentsResponse:
    service = _service(request)
    documents = await service.list_documents(collection_id)
    for document in documents:
        await service.delete_document(collection_id, document.id)
    return DeleteDocumentsResponse(deleted=len(documents))


@router.get(
    "/v1/rag/collections/{collection_id}/documents/{document_id}",
    response_model=DocumentResponse,
)
async def get_document(collection_id: str, document_id: str, request: Request) -> DocumentResponse:
    await _service(request).get_collection(collection_id)
    async with _state(request).db.session() as session:
        document = await session.scalar(
            select(RagDocument).where(
                RagDocument.id == document_id,
                RagDocument.collection_id == collection_id,
            )
        )
        if document is None:
            raise NotFoundError(f"Unknown RAG document: {document_id}")
        return DocumentResponse.of(document)


@router.delete(
    "/v1/rag/collections/{collection_id}/documents/{document_id}",
    dependencies=_WRITE,
    response_model=DeleteResponse,
)
async def delete_document_route(
    collection_id: str, document_id: str, request: Request
) -> DeleteResponse:
    await _service(request).delete_document(collection_id, document_id)
    return DeleteResponse(deleted=True)


@router.get("/v1/rag/collections/{collection_id}/chunks", response_model=ChunkListResponse)
async def list_chunks(
    collection_id: str,
    request: Request,
    document_id: str | None = Query(default=None),
    limit: int = Query(default=DEFAULT_CHUNK_PAGE_SIZE, gt=0, le=MAX_CHUNK_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    include_embeddings: bool = Query(default=False),
) -> ChunkListResponse:
    service = _service(request)
    chunks, total = await service.list_chunks(
        collection_id, document_id=document_id, limit=limit, offset=offset
    )
    vectors: dict[str, list[float]] = {}
    if include_embeddings and chunks:
        vectors = await service.chunk_vectors(collection_id, [chunk.id for chunk in chunks])
    return ChunkListResponse(
        collection_id=collection_id,
        document_id=document_id,
        total=total,
        limit=limit,
        offset=offset,
        items=[ChunkResponse.of(chunk, vectors.get(chunk.id)) for chunk in chunks],
    )


@router.get("/v1/rag/collections/{collection_id}/chunks/{chunk_id}", response_model=ChunkResponse)
async def get_chunk(
    collection_id: str,
    chunk_id: str,
    request: Request,
    include_embedding: bool = Query(default=True),
) -> ChunkResponse:
    service = _service(request)
    chunk = await service.get_chunk(collection_id, chunk_id)
    vector: list[float] | None = None
    if include_embedding:
        vector = (await service.chunk_vectors(collection_id, [chunk.id])).get(chunk.id)
    return ChunkResponse.of(chunk, vector)


@router.post("/v1/rag/search", response_model=SearchResponse)
async def search(payload: SearchRequest, request: Request) -> SearchResponse:
    chunks = await _service(request).search(
        payload.collection_id,
        payload.query,
        top_k=payload.top_k,
        min_score=payload.min_score,
        filters=payload.filters,
        diversity=payload.diversity,
    )
    return SearchResponse(collection_id=payload.collection_id, results=_chunk_response(chunks))


@router.post("/v1/rag/query", dependencies=_WRITE, response_model=RagQueryResponse)
async def query(
    payload: RagQueryRequest, request: Request, principal: CurrentPrincipal
) -> RagQueryResponse:
    service = _service(request)
    query_text = payload.query or next(
        (item.text() for item in reversed(payload.request.messages) if item.role.value == "user"),
        "",
    )
    chunks = await service.search(
        payload.collection_id,
        query_text,
        top_k=payload.top_k,
        min_score=payload.min_score,
        diversity=payload.diversity,
        filters=payload.filters,
    )
    context = build_context(chunks, max_tokens=payload.max_context_tokens)
    chat_request = augment_request(payload.request, context, mode=payload.mode)
    state = _state(request)
    pipeline = state.require_pipeline()
    # Run as the caller so their key's budget, limits, and model allowlist
    # apply. Console admins have no key of their own; like the playground,
    # they run with the master key.
    credential = principal.credential or state.settings.master_key.get_secret_value()
    chat_request = chat_request.model_copy(
        update={"stream": False, "metadata": {**chat_request.metadata, "api_key": credential}}
    )
    ctx = RequestContext(request=chat_request, state=state, route="/v1/rag/query")
    response = await pipeline.run(ctx)
    return RagQueryResponse(response=response, sources=_chunk_response(chunks))
