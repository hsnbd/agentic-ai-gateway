"""HTTP API for gateway-owned retrieval-augmented generation."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, Query, Request, Response, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sqlalchemy import select
from starlette.datastructures import UploadFile

from app.api.deps import CurrentPrincipal, require_gateway_writer
from app.core.errors import InvalidRequestError, NotFoundError
from app.core.pipeline import RequestContext
from app.core.schemas import ChatRequest, ChatResponse, RagOptions
from app.core.state import GatewayState
from app.db.models import RagChunk, RagDocument
from app.rag.access import RagAccess
from app.rag.extract import extract_text
from app.rag.retrieve import RetrievedChunk
from app.rag.service import DEFAULT_CHUNK_PAGE_SIZE, MAX_CHUNK_PAGE_SIZE, RagService

router = APIRouter()

#: Mutating and billable routes: console viewers are read-only.
_WRITE = [Depends(require_gateway_writer)]


def _access(principal: CurrentPrincipal) -> RagAccess:
    return RagAccess.for_principal(principal)


#: The caller's collection access (app/rag/access.py).
Access = Annotated[RagAccess, Depends(_access)]

#: Uploaded files may be this many times RAG_MAX_DOCUMENT_BYTES before extraction.
_UPLOAD_EXPANSION = 4

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
    #: Document metadata keys to index for search filters; fixed at creation.
    filterable_fields: list[str] = Field(default_factory=list)


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
    #: Owning team or key; both null for a global collection.
    owner_team_id: str | None = None
    owner_key_id: str | None = None
    filterable_fields: list[str] = Field(default_factory=list)
    created_at: datetime
    updated_at: datetime


class DocumentIngestRequest(BaseModel):
    content: str
    source: str = ""
    title: str | None = None
    content_type: str = "text/plain"
    metadata: dict[str, Any] = Field(default_factory=dict)
    #: Remove older documents from the same source once this one is ready.
    replace_existing: bool = True


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
    #: "vector" (embedding similarity) or "hybrid" (plus keyword search, fused by rank).
    search_mode: Literal["vector", "hybrid"] = "vector"
    #: Chat model that reorders the candidates by relevance; omitted skips reranking.
    rerank_model: str | None = None


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
    search_mode: Literal["vector", "hybrid"] = "vector"
    rerank_model: str | None = None
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
    # Created by GatewayState.startup, shared with the RAG pipeline stage.
    service: RagService = _state(request).components["rag_service"]
    return service


def _chunk_response(chunks: list[RetrievedChunk]) -> list[RetrievedChunkResponse]:
    return [RetrievedChunkResponse.model_validate(chunk) for chunk in chunks]


@router.post(
    "/v1/rag/collections",
    dependencies=_WRITE,
    response_model=CollectionResponse,
    status_code=status.HTTP_201_CREATED,
)
async def create_collection(
    payload: CollectionCreate, request: Request, access: Access
) -> CollectionResponse:
    service = _service(request)
    collection = await service.create_collection(
        payload.name,
        description=payload.description,
        embedding_model=payload.embedding_model,
        embedding_dimensions=payload.embedding_dimensions,
        chunk_size=payload.chunk_size,
        chunk_overlap=payload.chunk_overlap,
        metadata=payload.metadata,
        access=access,
        filterable_fields=payload.filterable_fields,
    )
    return CollectionResponse.model_validate(collection)


@router.get("/v1/rag/collections", response_model=list[CollectionResponse])
async def list_collections(request: Request, access: Access) -> list[CollectionResponse]:
    return [
        CollectionResponse.model_validate(item)
        for item in await _service(request).list_collections(access)
    ]


@router.get("/v1/rag/collections/{collection_id}", response_model=CollectionResponse)
async def get_collection(
    collection_id: str, request: Request, access: Access
) -> CollectionResponse:
    collection = await _service(request).get_collection(collection_id, access)
    return CollectionResponse.model_validate(collection)


@router.delete(
    "/v1/rag/collections/{collection_id}", dependencies=_WRITE, response_model=DeleteResponse
)
async def delete_collection(collection_id: str, request: Request, access: Access) -> DeleteResponse:
    await _service(request).delete_collection(collection_id, access)
    return DeleteResponse(deleted=True)


class IndexStatusResponse(BaseModel):
    collection_id: str
    index_exists: bool
    indexed_chunks: int
    stored_chunks: int
    in_sync: bool
    reindex: dict[str, Any] | None = None


@router.get("/v1/rag/collections/{collection_id}/index", response_model=IndexStatusResponse)
async def index_status(collection_id: str, request: Request, access: Access) -> IndexStatusResponse:
    return IndexStatusResponse(**await _service(request).index_status(collection_id, access))


@router.post(
    "/v1/rag/collections/{collection_id}/reindex",
    dependencies=_WRITE,
    response_model=IndexStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def reindex(
    collection_id: str, request: Request, access: Access, background: BackgroundTasks
) -> IndexStatusResponse:
    """Rebuild the collection's vectors from stored chunks; poll ``.../index`` for progress."""
    service = _service(request)
    await service.start_reindex(collection_id, access)
    background.add_task(service.reindex, collection_id)
    return IndexStatusResponse(**await service.index_status(collection_id, access))


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
async def ingest_document_route(
    collection_id: str, request: Request, access: Access, response: Response
) -> DocumentResponse:
    """Ingest JSON text or an uploaded file (.txt, .md, .html, .pdf, .docx).

    Large documents return 202 with status ``processing``; poll the document
    until it is ``ready`` or ``failed``.
    """
    service = _service(request)
    # Check before reading the body, so a refused upload is not parsed first.
    await service.get_collection(collection_id, access, write=True)
    settings = _state(request).settings
    content_type = request.headers.get("content-type", "").split(";", maxsplit=1)[0].lower()
    replace_existing = True
    if content_type == "application/json":
        payload = DocumentIngestRequest.model_validate(await request.json())
        content = payload.content
        source = payload.source
        replace_existing = payload.replace_existing
        metadata = {**payload.metadata, "content_type": payload.content_type}
        if payload.title:
            metadata["title"] = payload.title
    elif content_type == "multipart/form-data":
        form = await request.form()
        uploaded = form.get("file")
        if not isinstance(uploaded, UploadFile):
            raise InvalidRequestError("Multipart document ingestion requires a file field")
        filename = uploaded.filename or ""
        # Binary formats shrink when extracted; still bound what is read into memory.
        limit = settings.rag_max_document_bytes * _UPLOAD_EXPANSION
        data = await uploaded.read(limit + 1)
        if len(data) > limit:
            raise InvalidRequestError(f"Upload exceeds {limit} bytes", status_code=413)
        content, media_type = extract_text(data, filename)
        source = filename
        metadata = {"content_type": media_type}
        replace_existing = str(form.get("replace_existing", "true")).lower() != "false"
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

    document = await service.ingest(
        collection_id,
        content,
        source=source,
        metadata=metadata,
        access=access,
        replace_existing=replace_existing,
    )
    if document.status == "processing":
        response.status_code = status.HTTP_202_ACCEPTED
    return DocumentResponse.of(document)


@router.get(
    "/v1/rag/collections/{collection_id}/documents",
    response_model=list[DocumentResponse],
)
async def list_documents(
    collection_id: str, request: Request, access: Access
) -> list[DocumentResponse]:
    documents = await _service(request).list_documents(collection_id, access)
    return [DocumentResponse.of(item) for item in documents]


@router.delete(
    "/v1/rag/collections/{collection_id}/documents",
    dependencies=_WRITE,
    response_model=DeleteDocumentsResponse,
)
async def delete_documents(
    collection_id: str, request: Request, access: Access
) -> DeleteDocumentsResponse:
    service = _service(request)
    await service.get_collection(collection_id, access, write=True)
    documents = await service.list_documents(collection_id, access)
    for document in documents:
        await service.delete_document(collection_id, document.id, access)
    return DeleteDocumentsResponse(deleted=len(documents))


@router.get(
    "/v1/rag/collections/{collection_id}/documents/{document_id}",
    response_model=DocumentResponse,
)
async def get_document(
    collection_id: str, document_id: str, request: Request, access: Access
) -> DocumentResponse:
    await _service(request).get_collection(collection_id, access)
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
    collection_id: str, document_id: str, request: Request, access: Access
) -> DeleteResponse:
    await _service(request).delete_document(collection_id, document_id, access)
    return DeleteResponse(deleted=True)


@router.get("/v1/rag/collections/{collection_id}/chunks", response_model=ChunkListResponse)
async def list_chunks(
    collection_id: str,
    request: Request,
    access: Access,
    document_id: str | None = Query(default=None),
    limit: int = Query(default=DEFAULT_CHUNK_PAGE_SIZE, gt=0, le=MAX_CHUNK_PAGE_SIZE),
    offset: int = Query(default=0, ge=0),
    include_embeddings: bool = Query(default=False),
) -> ChunkListResponse:
    service = _service(request)
    chunks, total = await service.list_chunks(
        collection_id, document_id=document_id, limit=limit, offset=offset, access=access
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
    access: Access,
    include_embedding: bool = Query(default=True),
) -> ChunkResponse:
    service = _service(request)
    chunk = await service.get_chunk(collection_id, chunk_id, access)
    vector: list[float] | None = None
    if include_embedding:
        vector = (await service.chunk_vectors(collection_id, [chunk.id])).get(chunk.id)
    return ChunkResponse.of(chunk, vector)


@router.post("/v1/rag/search", response_model=SearchResponse)
async def search(payload: SearchRequest, request: Request, access: Access) -> SearchResponse:
    chunks = await _service(request).search(
        payload.collection_id,
        payload.query,
        top_k=payload.top_k,
        min_score=payload.min_score,
        filters=payload.filters,
        diversity=payload.diversity,
        search_mode=payload.search_mode,
        rerank_model=payload.rerank_model,
        access=access,
    )
    return SearchResponse(collection_id=payload.collection_id, results=_chunk_response(chunks))


@router.post("/v1/rag/query", dependencies=_WRITE, response_model=RagQueryResponse)
async def query(
    payload: RagQueryRequest, request: Request, principal: CurrentPrincipal
) -> RagQueryResponse:
    state = _state(request)
    pipeline = state.require_pipeline()
    # Retrieval runs inside the pipeline (RagStage), exactly as for a chat
    # request carrying `aigw.rag`, so both paths share one implementation.
    try:
        options = RagOptions(
            collection_id=payload.collection_id,
            query=payload.query,
            top_k=payload.top_k,
            min_score=payload.min_score,
            diversity=payload.diversity,
            filters=payload.filters,
            search_mode=payload.search_mode,
            rerank_model=payload.rerank_model,
            max_context_tokens=payload.max_context_tokens,
            mode=payload.mode,  # type: ignore[arg-type]
        )
    except ValidationError as exc:
        raise InvalidRequestError(f"Invalid RAG options: {exc}") from exc
    # Run as the caller so their key's budget, limits, and model allowlist
    # apply. Console admins have no key of their own; like the playground,
    # they run with the master key.
    credential = principal.credential or state.settings.master_key.get_secret_value()
    chat_request = payload.request.model_copy(
        update={
            "stream": False,
            "rag": options,
            "metadata": {**payload.request.metadata, "api_key": credential},
        }
    )
    ctx = RequestContext(request=chat_request, state=state, route="/v1/rag/query")
    response = await pipeline.run(ctx)
    return RagQueryResponse(
        response=response,
        sources=[RetrievedChunkResponse.model_validate(item) for item in ctx.rag_sources],
    )
