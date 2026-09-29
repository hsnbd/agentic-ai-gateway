import { useQuery } from '@tanstack/react-query';
import { apiRequest } from '../client';

export interface RagCollection {
  id: string;
  name: string;
  description: string | null;
  embedding_model: string;
  embedding_dimensions: number;
  chunk_size: number;
  chunk_overlap: number;
  document_count: number;
  chunk_count: number;
  created_at: string;
  updated_at: string;
}

export interface RagDocument {
  id: string;
  collection_id: string;
  title: string;
  source: string | null;
  content_type: string;
  content_hash: string;
  byte_size: number;
  status: string;
  chunk_count: number;
  error_message: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
  ingested_at: string | null;
}

export interface RagChunk {
  id: string;
  collection_id: string;
  document_id: string;
  chunk_index: number;
  text: string;
  token_count: number;
  char_count: number;
  vector_id: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  embedding: { dimensions: number; preview: number[]; truncated: boolean } | null;
}

export interface RagChunkPage {
  collection_id: string;
  document_id: string | null;
  total: number;
  limit: number;
  offset: number;
  items: RagChunk[];
}

export interface RagSearchResult {
  id: string;
  text: string;
  score: number;
  source: string | null;
  document_id: string | null;
  metadata: Record<string, unknown>;
}

export interface RagSearchResponse {
  collection_id: string;
  results: RagSearchResult[];
}

export const ragKeys = {
  all: ['rag'] as const,
  collections: ['rag', 'collections'] as const,
  documents: (collectionId: string) => ['rag', 'documents', collectionId] as const,
  chunks: (collectionId: string, documentId: string, limit: number, offset: number) => ['rag', 'chunks', collectionId, documentId, limit, offset] as const,
  chunk: (collectionId: string, chunkId: string) => ['rag', 'chunk', collectionId, chunkId] as const,
};

export function useRagCollections() {
  return useQuery({
    queryKey: ragKeys.collections,
    queryFn: () => apiRequest<RagCollection[]>('/v1/rag/collections'),
    staleTime: 30_000,
  });
}

export function useRagDocuments(collectionId: string | undefined) {
  return useQuery({
    queryKey: ragKeys.documents(collectionId ?? ''),
    queryFn: () => apiRequest<RagDocument[]>(`/v1/rag/collections/${encodeURIComponent(collectionId ?? '')}/documents`),
    enabled: Boolean(collectionId),
    staleTime: 15_000,
  });
}

export function useRagChunks(collectionId: string | undefined, documentId: string, limit: number, offset: number, enabled: boolean) {
  const pageSize = Math.max(1, Math.min(200, Math.trunc(limit) || 50));
  const params = new URLSearchParams({ limit: String(pageSize), offset: String(Math.max(0, offset)), include_embeddings: 'false' });
  if (documentId) params.set('document_id', documentId);
  return useQuery({
    queryKey: ragKeys.chunks(collectionId ?? '', documentId, pageSize, offset),
    queryFn: () => apiRequest<RagChunkPage>(`/v1/rag/collections/${encodeURIComponent(collectionId ?? '')}/chunks?${params}`),
    enabled: enabled && Boolean(collectionId),
    staleTime: 15_000,
  });
}

export function useRagChunk(collectionId: string | undefined, chunkId: string | undefined) {
  return useQuery({
    queryKey: ragKeys.chunk(collectionId ?? '', chunkId ?? ''),
    queryFn: () => apiRequest<RagChunk>(`/v1/rag/collections/${encodeURIComponent(collectionId ?? '')}/chunks/${encodeURIComponent(chunkId ?? '')}?include_embedding=true`),
    enabled: Boolean(collectionId && chunkId),
  });
}
