import { useState, type ChangeEvent, type FormEvent, type ReactNode } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { Alert, Box, Button, Card, CardContent, Checkbox, Chip, Dialog, DialogActions, DialogContent, DialogTitle, Divider, FormControlLabel, LinearProgress, MenuItem, Paper, Stack, Tab, Tabs, TextField, Typography } from '@mui/material';
import { apiRequest } from '../api/client';
import { ragKeys, useRagCollections, useRagDocuments, type RagCollection, type RagDocument, type RagSearchResponse, type RagSearchResult } from '../api/hooks/useRagCollections';
import { useAuth } from '../auth/AuthProvider';
import { ConfirmDialog, EmptyState, ErrorState, LoadingState, PageHeader } from '../components/Shared';
import { LongText } from '../features/knowledge/LongText';
import { ChunkInspector } from '../features/knowledge/ChunkInspector';

type RagTab = 'documents' | 'chunks' | 'retrieval';
type IngestMode = 'file' | 'text';
type ChunkingStrategy = 'recursive' | 'markdown';

function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString();
}

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function responseError(payload: unknown, fallback: string): string {
  if (payload instanceof Error && payload.message) return payload.message;
  if (typeof payload === 'object' && payload !== null) {
    const body = payload as { detail?: unknown; error?: { message?: unknown } };
    if (typeof body.error?.message === 'string') return body.error.message;
    if (typeof body.detail === 'string') return body.detail;
    if (typeof body.detail === 'object' && body.detail !== null && 'message' in body.detail) {
      const message = (body.detail as { message?: unknown }).message;
      if (typeof message === 'string') return message;
    }
  }
  return fallback;
}

function uploadWithProgress(collectionId: string, body: FormData, onProgress: (value: number) => void): Promise<RagDocument> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', `/v1/rag/collections/${encodeURIComponent(collectionId)}/documents`);
    const token = localStorage.getItem('aigateway.console.token');
    if (token) xhr.setRequestHeader('Authorization', `Bearer ${token}`);
    xhr.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable) onProgress(Math.round((event.loaded / event.total) * 100));
    });
    xhr.addEventListener('load', () => {
      let payload: unknown;
      try { payload = xhr.responseText ? JSON.parse(xhr.responseText) as unknown : null; }
      catch { reject(new Error(`Upload failed with status ${xhr.status}: response was not valid JSON`)); return; }
      if (xhr.status === 401) {
        localStorage.removeItem('aigateway.console.token');
        if (window.location.pathname !== '/ui/login') window.location.assign('/ui/login');
      }
      if (xhr.status < 200 || xhr.status >= 300) {
        reject(new Error(responseError(payload, `Upload failed with status ${xhr.status}`)));
        return;
      }
      resolve(payload as RagDocument);
    });
    xhr.addEventListener('error', () => reject(new Error('Upload failed because the network connection was interrupted.')));
    xhr.addEventListener('abort', () => reject(new Error('Upload was cancelled.')));
    xhr.send(body);
  });
}

function HighlightedText({ text, query }: { text: string; query: string }) {
  const [expanded, setExpanded] = useState(false);
  const terms = query.trim().split(/\s+/).filter(Boolean).sort((a, b) => b.length - a.length);
  if (!terms.length) return <LongText text={text} />;
  const pattern = new RegExp(`(${terms.map((term) => term.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})`, 'ig');
  const parts = text.split(pattern);
  return <Box sx={{ overflowWrap: 'anywhere' }}>
    <Typography variant="body2" sx={expanded ? { whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' } : { whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', overflow: 'hidden', display: '-webkit-box', WebkitBoxOrient: 'vertical', WebkitLineClamp: 5 }}>{parts.map((part, index) =>
      terms.some((term) => term.toLocaleLowerCase() === part.toLocaleLowerCase())
        ? <Box key={`${part}-${index}`} component="mark" sx={{ bgcolor: 'warning.light', color: 'text.primary', px: 0.2 }}>{part}</Box>
        : part,
    )}</Typography>
    {text.length > 240 && <Button size="small" onClick={() => setExpanded((value) => !value)} sx={{ px: 0, minWidth: 0 }}>{expanded ? 'Show less' : 'Expand'}</Button>}
  </Box>;
}

function Metric({ label, value }: { label: string; value: ReactNode }) {
  return <Box><Typography variant="caption" color="text.secondary">{label}</Typography><Typography variant="body2" sx={{ fontWeight: 600, overflowWrap: 'anywhere' }}>{value}</Typography></Box>;
}

export default function RagCollections() {
  const { user } = useAuth();
  const canManage = user?.role === 'admin';
  const queryClient = useQueryClient();
  const collectionsQuery = useRagCollections();
  const [selectedId, setSelectedId] = useState<string | undefined>();
  const collections = [...(collectionsQuery.data ?? [])].sort((a, b) => (collectionSort === 'newest' ? -1 : 1) * (Date.parse(a.created_at) - Date.parse(b.created_at)));
  const selected = collections.find((item) => item.id === selectedId) ?? collections[0];
  const documentsQuery = useRagDocuments(selected?.id);
  const [tab, setTab] = useState<RagTab>('documents');
  const [createOpen, setCreateOpen] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<RagCollection | null>(null);
  const [documentToDelete, setDocumentToDelete] = useState<RagDocument | null>(null);
  const [retryTarget, setRetryTarget] = useState<RagDocument | null>(null);
  const [retryFile, setRetryFile] = useState<File | null>(null);
  const [retryText, setRetryText] = useState('');
  const [retryError, setRetryError] = useState<string | null>(null);
  const [retryPending, setRetryPending] = useState(false);
  const [documentSort, setDocumentSort] = useState<'newest' | 'oldest'>('newest');
  const [collectionSort, setCollectionSort] = useState<'newest' | 'oldest'>('newest');
  const [formError, setFormError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState<number | null>(null);
  const [ingestMode, setIngestMode] = useState<IngestMode>('file');
  const [chunkingStrategy, setChunkingStrategy] = useState<ChunkingStrategy>('recursive');
  const [searchQuery, setSearchQuery] = useState('');
  const [topK, setTopK] = useState(5);
  const [minScore, setMinScore] = useState(0);
  const [useScoreThreshold, setUseScoreThreshold] = useState(false);
  const [searchResults, setSearchResults] = useState<RagSearchResult[] | null>(null);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [newCollection, setNewCollection] = useState({ name: '', description: '', embeddingModel: '', chunkSize: 1000, chunkOverlap: 150 });

  const createMutation = useMutation({
    mutationFn: () => apiRequest<RagCollection>('/v1/rag/collections', {
      method: 'POST',
      body: {
        name: newCollection.name.trim(),
        description: newCollection.description.trim() || null,
        ...(newCollection.embeddingModel.trim() ? { embedding_model: newCollection.embeddingModel.trim() } : {}),
        chunk_size: newCollection.chunkSize,
        chunk_overlap: newCollection.chunkOverlap,
      },
    }),
    onSuccess: async (created) => {
      await queryClient.invalidateQueries({ queryKey: ragKeys.collections });
      setSelectedId(created.id);
      setTab('documents');
      setCreateOpen(false);
      setNewCollection({ name: '', description: '', embeddingModel: '', chunkSize: 1000, chunkOverlap: 150 });
      setFormError(null);
    },
    onError: (error: Error) => setFormError(error.message),
  });

  const deleteCollectionMutation = useMutation({
    mutationFn: (collectionId: string) => apiRequest<{ deleted: boolean }>(`/v1/rag/collections/${encodeURIComponent(collectionId)}`, { method: 'DELETE' }),
    onSuccess: async (_, collectionId) => {
      await queryClient.invalidateQueries({ queryKey: ragKeys.collections });
      setSelectedId((current) => current === collectionId ? undefined : current);
      setDeleteTarget(null);
    },
  });

  const deleteDocumentMutation = useMutation({
    mutationFn: (documentId: string) => apiRequest<{ deleted: boolean }>(`/v1/rag/collections/${encodeURIComponent(selected?.id ?? '')}/documents/${encodeURIComponent(documentId)}`, { method: 'DELETE' }),
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ragKeys.documents(selected?.id ?? '') }),
        queryClient.invalidateQueries({ queryKey: ragKeys.collections }),
        queryClient.invalidateQueries({ queryKey: ['rag', 'chunks', selected?.id ?? ''] }),
      ]);
      setDocumentToDelete(null);
    },
  });

  const orderedDocuments = [...(documentsQuery.data ?? [])].sort((a, b) => (documentSort === 'newest' ? -1 : 1) * (Date.parse(a.created_at) - Date.parse(b.created_at)));

  const submitCreate = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setFormError(null);
    createMutation.mutate();
  };

  const uploadFile = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = '';
    if (!file || !selected) return;
    setUploadError(null);
    setUploadProgress(0);
    const body = new FormData();
    body.append('file', file);
    body.append('metadata', JSON.stringify({ chunking_strategy: chunkingStrategy }));
    try {
      await uploadWithProgress(selected.id, body, (value) => setUploadProgress(value === 100 ? -2 : value));
      setUploadProgress(100);
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ragKeys.documents(selected.id) }),
        queryClient.invalidateQueries({ queryKey: ragKeys.collections }),
        queryClient.invalidateQueries({ queryKey: ['rag', 'chunks', selected.id] }),
      ]);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Document upload failed.');
    } finally {
      window.setTimeout(() => setUploadProgress(null), 900);
    }
  };

  const uploadText = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selected) return;
    const formElement = event.currentTarget;
    const form = new FormData(formElement);
    const content = String(form.get('content') ?? '');
    const title = String(form.get('title') ?? '').trim();
    setUploadError(null);
    setUploadProgress(-1);
    try {
      await apiRequest<RagDocument>(`/v1/rag/collections/${encodeURIComponent(selected.id)}/documents`, {
        method: 'POST',
        body: { content, source: title, title: title || null, content_type: 'text/plain', metadata: { chunking_strategy: chunkingStrategy } },
      });
      setUploadProgress(100);
      formElement.reset();
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ragKeys.documents(selected.id) }),
        queryClient.invalidateQueries({ queryKey: ragKeys.collections }),
        queryClient.invalidateQueries({ queryKey: ['rag', 'chunks', selected.id] }),
      ]);
    } catch (error) {
      setUploadError(error instanceof Error ? error.message : 'Text ingestion failed.');
    } finally {
      window.setTimeout(() => setUploadProgress(null), 900);
    }
  };

  const retryDocument = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!retryTarget) return;
    const collectionId = retryTarget.collection_id;
    setRetryError(null);
    setRetryPending(true);
    try {
      const content = retryFile ? await retryFile.text() : retryText;
      if (!content) throw new Error('Provide the original file or paste the original text to retry.');
      const digest = await crypto.subtle.digest('SHA-256', new TextEncoder().encode(content));
      const hash = Array.from(new Uint8Array(digest), (byte) => byte.toString(16).padStart(2, '0')).join('');
      if (hash !== retryTarget.content_hash) throw new Error('Content does not match the failed document. Select the original file or paste its exact text.');
      if (retryFile) {
        const body = new FormData();
        body.append('file', retryFile);
        body.append('metadata', JSON.stringify({ chunking_strategy: retryTarget.metadata.chunking_strategy ?? 'recursive' }));
        await uploadWithProgress(collectionId, body, () => undefined);
      } else {
        await apiRequest<RagDocument>(`/v1/rag/collections/${encodeURIComponent(collectionId)}/documents`, {
          method: 'POST', body: { content, source: retryTarget.source ?? '', title: retryTarget.title, content_type: retryTarget.content_type, metadata: retryTarget.metadata },
        });
      }
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ragKeys.documents(collectionId) }),
        queryClient.invalidateQueries({ queryKey: ragKeys.collections }),
        queryClient.invalidateQueries({ queryKey: ['rag', 'chunks', collectionId] }),
      ]);
      setRetryTarget(null);
      setRetryFile(null);
      setRetryText('');
    } catch (error) {
      setRetryError(error instanceof Error ? error.message : 'Retry failed.');
    } finally {
      setRetryPending(false);
    }
  };

  const runSearch = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!selected || !searchQuery.trim()) return;
    setSearchLoading(true);
    setSearchError(null);
    setSearchResults(null);
    try {
      const response = await apiRequest<RagSearchResponse>('/v1/rag/search', {
        method: 'POST',
        body: { collection_id: selected.id, query: searchQuery.trim(), top_k: topK, ...(useScoreThreshold ? { min_score: minScore } : {}) },
      });
      setSearchResults(response.results);
    } catch (error) {
      setSearchError(error instanceof Error ? error.message : 'Retrieval request failed.');
    } finally {
      setSearchLoading(false);
    }
  };

  const headerAction = canManage ? <Button variant="contained" onClick={() => { setFormError(null); setCreateOpen(true); }}>Create collection</Button> : null;

  return <Box>
    <PageHeader title="RAG collections" description="Manage indexed knowledge, inspect retrieval results, and test score thresholds." action={headerAction} />
    <TextField select size="small" label="Collections by recency" value={collectionSort} onChange={(event) => setCollectionSort(event.target.value === 'oldest' ? 'oldest' : 'newest')} sx={{ mb: 2, minWidth: 190 }}><MenuItem value="newest">Newest first</MenuItem><MenuItem value="oldest">Oldest first</MenuItem></TextField>
    {collectionsQuery.isLoading ? <LoadingState label="Loading collections" /> : collectionsQuery.isError ? <ErrorState error={collectionsQuery.error} onRetry={() => void collectionsQuery.refetch()} /> : collections.length === 0 ?
      <Paper><EmptyState title="No collections yet" description={canManage ? 'Create a collection to start ingesting documents and testing retrieval.' : 'No retrieval collections are available.'} /></Paper> :
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'minmax(260px, 0.85fr) minmax(0, 2.15fr)' }, gap: 2, alignItems: 'start' }}>
        <Stack spacing={1.5}>
          {collections.map((collection) => <Card key={collection.id} variant="outlined" sx={{ borderColor: collection.id === selected?.id ? 'primary.main' : 'divider' }}>
            <CardContent sx={{ '&:last-child': { pb: 2 } }}>
              <Stack direction="row" justifyContent="space-between" alignItems="flex-start" spacing={1}>
                <Button onClick={() => { setSelectedId(collection.id); setSearchResults(null); }} sx={{ p: 0, minWidth: 0, textAlign: 'left', justifyContent: 'flex-start', fontSize: '1rem', fontWeight: 700, overflowWrap: 'anywhere' }}>{collection.name}</Button>
                {canManage && <Button size="small" color="error" onClick={() => setDeleteTarget(collection)}>Delete</Button>}
              </Stack>
              <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>{collection.description || 'No description'}</Typography>
              <Stack direction="row" gap={1} flexWrap="wrap" sx={{ mt: 1.5 }}>
                <Chip size="small" label={`${collection.document_count} documents`} />
                <Chip size="small" label={`${collection.chunk_count} chunks`} />
              </Stack>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1.5, overflowWrap: 'anywhere' }}>Embedding: {collection.embedding_model}</Typography>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>Created: {formatDate(collection.created_at)} · Updated: {formatDate(collection.updated_at)}</Typography>
            </CardContent>
          </Card>)}
        </Stack>

        {selected && <Paper sx={{ minWidth: 0, overflow: 'hidden' }}>
          <Box sx={{ p: { xs: 2, md: 3 }, pb: 1 }}>
            <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" gap={2}>
              <Box sx={{ minWidth: 0 }}><Typography variant="h2" sx={{ overflowWrap: 'anywhere' }}>{selected.name}</Typography><Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>{selected.embedding_model} · {selected.embedding_dimensions} dimensions</Typography></Box>
              <Stack direction="row" gap={1} flexWrap="wrap"><Chip label={`Chunk size ${selected.chunk_size} chars`} /><Chip label={`Overlap ${selected.chunk_overlap} chars`} /></Stack>
            </Stack>
          </Box>
          <Tabs value={tab} onChange={(_, value: RagTab) => setTab(value)} variant="scrollable" scrollButtons="auto" aria-label="Collection details">
            <Tab value="documents" label="Documents" />
            <Tab value="chunks" label="Chunks" />
            <Tab value="retrieval" label="Retrieval test" />
          </Tabs>
          <Divider />
          <Box sx={{ p: { xs: 2, md: 3 }, minWidth: 0 }}>
            {tab === 'documents' && <Stack spacing={2}>
              {canManage && <Card variant="outlined"><CardContent>
                <Typography variant="h3">Add documents</Typography>
                <Tabs value={ingestMode} onChange={(_, value: IngestMode) => setIngestMode(value)} sx={{ mt: 1 }} aria-label="Document input mode">
                  <Tab value="file" label="Upload file" />
                  <Tab value="text" label="Paste text" />
                </Tabs>
                <Stack direction={{ xs: 'column', sm: 'row' }} gap={1.5} sx={{ mt: 2 }}>
                  <TextField select label="Chunking strategy" size="small" value={chunkingStrategy} onChange={(event) => setChunkingStrategy(event.target.value === 'markdown' ? 'markdown' : 'recursive')} sx={{ minWidth: 190 }}>
                    <MenuItem value="recursive">Recursive</MenuItem><MenuItem value="markdown">Markdown sections</MenuItem>
                  </TextField>
                  <TextField label="Chunk size (characters)" size="small" value={selected.chunk_size} disabled />
                  <TextField label="Overlap (characters)" size="small" value={selected.chunk_overlap} disabled />
                </Stack>
                {ingestMode === 'file' ? <Stack spacing={1.5} sx={{ mt: 2 }}>
                  <Typography variant="body2" color="text.secondary">Supported formats: .txt and .md. Chunk size and overlap are configured on the collection; strategy is selected for each document.</Typography>
                  <Button component="label" variant="outlined" disabled={uploadProgress !== null}>Choose .txt or .md file<input hidden type="file" accept=".txt,.md,text/plain,text/markdown" onChange={(event) => void uploadFile(event)} /></Button>
                </Stack> : <Box component="form" onSubmit={(event: FormEvent<HTMLFormElement>) => void uploadText(event)} sx={{ display: 'grid', gap: 1.5, mt: 2 }}>
                  <TextField label="Document title / source" name="title" size="small" />
                  <TextField label="Document text" name="content" multiline minRows={5} required />
                  <Button type="submit" variant="contained" disabled={uploadProgress !== null}>Ingest text</Button>
                </Box>}
                {uploadProgress !== null && <Box sx={{ mt: 2 }}><Typography variant="caption" color="text.secondary">{uploadProgress === -2 ? 'Upload complete; processing document…' : uploadProgress === -1 ? 'Ingesting pasted text…' : uploadProgress === 100 ? 'Ingestion complete' : `Uploading file ${uploadProgress}%`}</Typography><LinearProgress variant={uploadProgress < 0 ? 'indeterminate' : 'determinate'} value={uploadProgress < 0 ? undefined : uploadProgress} sx={{ mt: 0.5 }} /></Box>}
                {uploadError && <Alert severity="error" sx={{ mt: 2 }}>{uploadError}</Alert>}
              </CardContent></Card>}
              <TextField select size="small" label="Documents by recency" value={documentSort} onChange={(event) => setDocumentSort(event.target.value === 'oldest' ? 'oldest' : 'newest')} sx={{ maxWidth: 190 }}><MenuItem value="newest">Newest first</MenuItem><MenuItem value="oldest">Oldest first</MenuItem></TextField>
              {documentsQuery.isLoading ? <LoadingState label="Loading documents" /> : documentsQuery.isError ? <ErrorState error={documentsQuery.error} onRetry={() => void documentsQuery.refetch()} /> : orderedDocuments.length === 0 ?
                <Paper variant="outlined"><EmptyState title="No documents in this collection" description={canManage ? 'Upload a .txt or .md file, or paste raw text above.' : 'This collection has no indexed documents.'} /></Paper> :
                <Stack spacing={1.25}>
                  {orderedDocuments.map((document) => <Card key={document.id} variant="outlined"><CardContent>
                    <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" alignItems={{ sm: 'flex-start' }} gap={1.5}>
                      <Box sx={{ minWidth: 0, flex: 1 }}><Stack direction="row" gap={1} alignItems="center" flexWrap="wrap"><Typography variant="h3" sx={{ overflowWrap: 'anywhere' }}>{document.title}</Typography><Chip size="small" label={document.status} color={document.status === 'ready' || document.status === 'completed' ? 'success' : document.status === 'failed' ? 'error' : 'default'} /></Stack>
                        <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5, overflowWrap: 'anywhere' }}>{document.source || 'Source not provided'} · {document.content_type}</Typography>
                        <Stack direction="row" gap={2} flexWrap="wrap" sx={{ mt: 1 }}><Metric label="Size" value={formatBytes(document.byte_size)} /><Metric label="Chunks" value={document.chunk_count} /><Metric label="Created" value={formatDate(document.created_at)} /><Metric label="Updated" value={formatDate(document.updated_at)} /><Metric label="Ingested" value={document.ingested_at ? formatDate(document.ingested_at) : 'Never finished'} /></Stack>
                        {document.status === 'failed' && document.error_message && <Alert severity="error" sx={{ mt: 1.5 }}>{document.error_message}</Alert>}
                        {document.status === 'failed' && !document.error_message && <Alert severity="error" sx={{ mt: 1.5 }}>Ingestion failed without an error message.</Alert>}
                      </Box>
                      {canManage && <Stack direction="row" gap={1}>{document.status === 'failed' && <Button onClick={() => { setRetryTarget(document); setRetryFile(null); setRetryText(''); setRetryError(null); }}>Retry</Button>}<Button color="error" onClick={() => setDocumentToDelete(document)} disabled={deleteDocumentMutation.isPending}>Delete</Button></Stack>}
                    </Stack>
                  </CardContent></Card>)}
                </Stack>}
              {deleteDocumentMutation.isError && <ErrorState error={deleteDocumentMutation.error} />}
            </Stack>}

            {tab === 'chunks' && <ChunkInspector key={selected.id} collectionId={selected.id} documents={documentsQuery.data ?? []} />}

            {tab === 'retrieval' && <Stack spacing={2}>
              <Card variant="outlined"><CardContent component="form" onSubmit={(event: FormEvent<HTMLFormElement>) => void runSearch(event)}>
                <Typography variant="h3">Test retrieval</Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>Search results are ranked by similarity; the API applies the optional minimum score before returning them.</Typography>
                <TextField label="Query" value={searchQuery} onChange={(event) => setSearchQuery(event.target.value)} fullWidth multiline minRows={2} required sx={{ mt: 2 }} />
                <Stack direction={{ xs: 'column', sm: 'row' }} gap={1.5} sx={{ mt: 1.5, alignItems: { sm: 'center' } }}>
                  <TextField label="Top-k" type="number" size="small" value={topK} onChange={(event) => setTopK(Math.max(1, Math.min(100, Number(event.target.value))))} inputProps={{ min: 1, max: 100 }} sx={{ width: { sm: 120 } }} />
                  <FormControlLabel control={<Checkbox checked={useScoreThreshold} onChange={(event) => setUseScoreThreshold(event.target.checked)} />} label="Set score threshold" />
                  <TextField label="Minimum score" type="number" size="small" value={minScore} onChange={(event) => setMinScore(Math.max(-1, Math.min(1, Number(event.target.value))))} inputProps={{ min: -1, max: 1, step: 0.05 }} helperText="-1 to 1" disabled={!useScoreThreshold} sx={{ width: { sm: 170 } }} />
                  <Button type="submit" variant="contained" disabled={searchLoading || !searchQuery.trim()} sx={{ ml: { sm: 'auto' } }}>Search collection</Button>
                </Stack>
              </CardContent></Card>
              {searchLoading && <LoadingState label="Searching and comparing similarity scores" />}
              {searchError && <ErrorState error={searchError} />}
              {searchResults !== null && (searchResults.length === 0 ? <Paper variant="outlined"><EmptyState title={useScoreThreshold ? 'No results crossed the score threshold' : 'No matching chunks found'} description={useScoreThreshold ? `Nothing met the minimum similarity score of ${minScore.toFixed(2)}. Lower the threshold or try another query.` : 'Try a different query, or ingest documents into this collection.'} /></Paper> :
                <Stack spacing={1.5}>
                  {searchResults.map((result, index) => <Card key={result.id} variant="outlined"><CardContent>
                    <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" gap={1}>
                      <Stack direction="row" alignItems="center" gap={1}><Chip label={`#${index + 1}`} size="small" /><Typography variant="h3">Similarity {(result.score * 100).toFixed(1)}%</Typography></Stack>
                      <Typography variant="body2" color="text.secondary" sx={{ fontVariantNumeric: 'tabular-nums' }}>{result.score.toFixed(4)} / 1.0000</Typography>
                    </Stack>
                    <LinearProgress variant="determinate" value={Math.max(0, Math.min(100, result.score * 100))} sx={{ height: 8, borderRadius: 5, my: 1.5 }} />
                    <HighlightedText text={result.text} query={searchQuery} />
                    <Divider sx={{ my: 1.5 }} />
                    <Stack direction="row" gap={2} flexWrap="wrap"><Metric label="Source document" value={result.source || result.document_id || 'Source unavailable'} /><Metric label="Document ID" value={result.document_id || 'Not provided'} /></Stack>
                  </CardContent></Card>)}
                </Stack>)}
            </Stack>}
          </Box>
        </Paper>}
      </Box>}

    <Dialog open={createOpen} onClose={() => setCreateOpen(false)} fullWidth maxWidth="sm">
      <Box component="form" onSubmit={submitCreate}>
        <DialogTitle>Create a RAG collection</DialogTitle>
        <DialogContent sx={{ display: 'grid', gap: 2, pt: '12px !important' }}>
          <TextField label="Name" required autoFocus value={newCollection.name} onChange={(event) => setNewCollection((current) => ({ ...current, name: event.target.value }))} />
          <TextField label="Description" multiline minRows={2} value={newCollection.description} onChange={(event) => setNewCollection((current) => ({ ...current, description: event.target.value }))} />
          <TextField label="Embedding model (optional)" helperText="Leave empty to use the gateway default." value={newCollection.embeddingModel} onChange={(event) => setNewCollection((current) => ({ ...current, embeddingModel: event.target.value }))} />
          <Stack direction={{ xs: 'column', sm: 'row' }} gap={2}>
            <TextField label="Chunk size (characters)" type="number" required value={newCollection.chunkSize} onChange={(event) => setNewCollection((current) => ({ ...current, chunkSize: Number(event.target.value) }))} inputProps={{ min: 1 }} />
            <TextField label="Overlap (characters)" type="number" required value={newCollection.chunkOverlap} onChange={(event) => setNewCollection((current) => ({ ...current, chunkOverlap: Number(event.target.value) }))} inputProps={{ min: 0 }} />
          </Stack>
          {newCollection.chunkSize > 0 && newCollection.chunkOverlap >= newCollection.chunkSize && <Alert severity="error">Overlap must be smaller than chunk size.</Alert>}
          {formError && <Alert severity="error">{formError}</Alert>}
          {createMutation.isError && formError === null && <ErrorState error={createMutation.error} />}
        </DialogContent>
        <DialogActions><Button onClick={() => setCreateOpen(false)}>Cancel</Button><Button type="submit" variant="contained" disabled={createMutation.isPending || !newCollection.name.trim() || newCollection.chunkSize <= 0 || newCollection.chunkOverlap < 0 || newCollection.chunkOverlap >= newCollection.chunkSize}>Create collection</Button></DialogActions>
      </Box>
    </Dialog>
    <ConfirmDialog open={Boolean(deleteTarget)} title="Delete collection?" description={`Deleting “${deleteTarget?.name ?? ''}” will permanently destroy all of its documents and embeddings. This cannot be undone.`} confirmLabel={deleteCollectionMutation.isPending ? 'Deleting…' : 'Delete collection'} onClose={() => setDeleteTarget(null)} onConfirm={() => { if (deleteTarget) deleteCollectionMutation.mutate(deleteTarget.id); }} />
    {deleteCollectionMutation.isError && <Alert severity="error" sx={{ mt: 2 }}>{deleteCollectionMutation.error.message}</Alert>}
    <Dialog open={Boolean(retryTarget)} onClose={() => { if (!retryPending) setRetryTarget(null); }} fullWidth maxWidth="sm">
      <Box component="form" onSubmit={(event: FormEvent<HTMLFormElement>) => void retryDocument(event)}>
        <DialogTitle>Retry failed document</DialogTitle>
        <DialogContent sx={{ display: 'grid', gap: 2, pt: '12px !important' }}>
          <Typography variant="body2">Supply the exact original content for “{retryTarget?.title}”. The gateway stores a content hash, not the original text; retry will verify the hash before ingestion.</Typography>
          <Button component="label" variant="outlined">Choose original .txt or .md file<input hidden type="file" accept=".txt,.md" onChange={(event) => { setRetryFile(event.target.files?.[0] ?? null); setRetryText(''); }} /></Button>
          {retryFile && <Typography variant="body2">Selected: {retryFile.name}</Typography>}
          <TextField label="Or paste original text" multiline minRows={4} value={retryText} disabled={Boolean(retryFile)} onChange={(event) => setRetryText(event.target.value)} />
          {retryError && <Alert severity="error">{retryError}</Alert>}
        </DialogContent>
        <DialogActions><Button onClick={() => setRetryTarget(null)} disabled={retryPending}>Cancel</Button><Button type="submit" variant="contained" disabled={retryPending || (!retryFile && !retryText)}>{retryPending ? 'Retrying…' : 'Retry ingestion'}</Button></DialogActions>
      </Box>
    </Dialog>
    <ConfirmDialog open={Boolean(documentToDelete)} title="Delete document?" description={`Permanently delete “${documentToDelete?.title ?? ''}” and its indexed chunks?`} confirmLabel={deleteDocumentMutation.isPending ? 'Deleting…' : 'Delete document'} onClose={() => setDocumentToDelete(null)} onConfirm={() => { if (documentToDelete) deleteDocumentMutation.mutate(documentToDelete.id); }} />
    {!(collectionsQuery.isLoading || collectionsQuery.isError || collections.length === 0) && !selected && <Alert severity="info">Select a collection to view its contents.</Alert>}
  </Box>;
}
