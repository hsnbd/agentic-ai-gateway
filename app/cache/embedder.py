"""Embedding client and bounded in-process embedding cache for semantic caching."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections import OrderedDict

from app.config.settings import Settings
from app.core.schemas import EmbeddingRequest
from app.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)


class CacheEmbedder:
    """Embeds cache queries through a configured embedding deployment."""

    def __init__(self, registry: ProviderRegistry, settings: Settings) -> None:
        self._registry = registry
        self._settings = settings
        self._dimensions: int | None = None
        self._embeddings: OrderedDict[str, list[float]] = OrderedDict()
        self._lock = asyncio.Lock()
        self._cache_size = 512

    @property
    def dimensions(self) -> int | None:
        """Embedding dimensions, learned from the first successful vector."""
        return self._dimensions

    async def embed(self, text: str) -> list[float] | None:
        """Embed one string, returning None rather than failing a gateway request."""
        vectors = await self.embed_batch([text])
        return vectors[0] if vectors is not None else None

    async def embed_batch(self, texts: list[str]) -> list[list[float]] | None:
        """Embed strings using the registry, reusing exact-text LRU entries."""
        if not texts:
            return []

        try:
            async with self._lock:
                hashes = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in texts]
                vectors: dict[str, list[float]] = {}
                missing: dict[str, str] = {}
                for text, text_hash in zip(texts, hashes, strict=True):
                    cached = self._embeddings.get(text_hash)
                    if cached is not None:
                        self._embeddings.move_to_end(text_hash)
                        vectors[text_hash] = cached
                    else:
                        missing.setdefault(text_hash, text)

                if missing:
                    embedded = await self._embed_uncached(list(missing.values()))
                    if embedded is None:
                        return None
                    for text_hash, vector in zip(missing, embedded, strict=True):
                        vectors[text_hash] = vector
                        self._embeddings[text_hash] = vector
                        self._embeddings.move_to_end(text_hash)
                    while len(self._embeddings) > self._cache_size:
                        self._embeddings.popitem(last=False)

                return [vectors[text_hash] for text_hash in hashes]
        except Exception:
            logger.warning("Cache embedding failed; skipping semantic cache", exc_info=True)
            return None

    async def _embed_uncached(self, texts: list[str]) -> list[list[float]] | None:
        try:
            deployments = self._registry.deployments_for(self._settings.cache_embedding_model)
            candidates = [
                deployment for deployment in deployments if deployment.capabilities.embeddings
            ]
            if not candidates:
                logger.warning(
                    "No embedding-capable deployment configured for cache model %r",
                    self._settings.cache_embedding_model,
                )
                return None

            request = EmbeddingRequest(
                model=self._settings.cache_embedding_model,
                input=texts,
                dimensions=self._settings.cache_embedding_dimensions,
            )
            last_error: Exception | None = None
            for deployment in candidates:
                try:
                    response = await self._registry.provider_for(deployment).embed(
                        request, deployment
                    )
                    indexed = sorted(response.data, key=lambda item: item.index)
                    if len(indexed) != len(texts):
                        raise ValueError("Embedding provider returned an unexpected vector count")
                    vectors = [[float(value) for value in item.embedding] for item in indexed]
                    dimension = len(vectors[0])
                    if dimension == 0 or any(len(vector) != dimension for vector in vectors):
                        raise ValueError("Embedding provider returned inconsistent dimensions")
                    if self._dimensions is not None and self._dimensions != dimension:
                        raise ValueError("Embedding dimensions changed after initial discovery")
                    self._dimensions = dimension
                    return vectors
                except Exception as exc:
                    last_error = exc

            logger.warning(
                "Unable to embed cache query with model %r; skipping semantic cache: %s",
                self._settings.cache_embedding_model,
                last_error,
            )
            return None
        except Exception:
            logger.warning("Unable to resolve cache embedding deployment", exc_info=True)
            return None
