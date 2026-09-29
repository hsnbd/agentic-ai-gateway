import { useState } from 'react';
import { Alert, Box, Button, Card, CardContent, MenuItem, Stack, TextField, Typography } from '@mui/material';
import { useRagChunk, useRagChunks, type RagDocument } from '../../api/hooks/useRagCollections';
import { EmptyState, ErrorState, JsonViewer, LoadingState } from '../../components/Shared';
import { LongText } from './LongText';

export function ChunkInspector({ collectionId, documents }: { collectionId: string; documents: RagDocument[] }) {
  const [documentId, setDocumentId] = useState('');
  const [limit, setLimit] = useState(50);
  const [offset, setOffset] = useState(0);
  const [openedId, setOpenedId] = useState<string | undefined>();
  const page = useRagChunks(collectionId, documentId, limit, offset, true);
  const detail = useRagChunk(collectionId, openedId);
  const documentNames = new Map(documents.map((document) => [document.id, document.title]));
  return <Stack spacing={2}>
    <Stack direction={{ xs: 'column', sm: 'row' }} gap={1.5}>
      <TextField select size="small" label="Source document" value={documentId} onChange={(event) => { setDocumentId(event.target.value); setOffset(0); setOpenedId(undefined); }} sx={{ minWidth: 200, flex: 1 }}>
        <MenuItem value="">All documents</MenuItem>
        {documents.map((document) => <MenuItem key={document.id} value={document.id}>{document.title}</MenuItem>)}
      </TextField>
      <TextField select size="small" label="Page size" value={limit} onChange={(event) => { setLimit(Math.max(1, Math.min(200, Number(event.target.value)))); setOffset(0); setOpenedId(undefined); }} sx={{ minWidth: 110 }}>
        {[25, 50, 100, 200].map((size) => <MenuItem value={size} key={size}>{size}</MenuItem>)}
      </TextField>
    </Stack>
    {page.isLoading ? <LoadingState label="Loading chunks" /> : page.isError ? <ErrorState error={page.error} onRetry={() => void page.refetch()} /> : page.data?.items.length === 0 ? <EmptyState title="No chunks found" description="Select another document or ingest content into this collection." /> : <>
      <Typography variant="body2" color="text.secondary">{page.data?.total ?? 0} chunks · Showing {(page.data?.offset ?? 0) + 1}–{Math.min((page.data?.offset ?? 0) + (page.data?.items.length ?? 0), page.data?.total ?? 0)}</Typography>
      {page.data?.items.map((chunk) => <Card key={chunk.id} variant="outlined"><CardContent sx={{ minWidth: 0 }}>
        <Stack direction={{ xs: 'column', sm: 'row' }} justifyContent="space-between" gap={1}>
          <Box sx={{ minWidth: 0 }}><Typography variant="subtitle2">Chunk #{chunk.chunk_index} · {documentNames.get(chunk.document_id) ?? chunk.document_id}</Typography><Typography variant="caption" color="text.secondary">{chunk.token_count} tokens · {chunk.char_count} characters</Typography></Box>
          <Button size="small" onClick={() => setOpenedId(openedId === chunk.id ? undefined : chunk.id)}>{openedId === chunk.id ? 'Close details' : 'View details'}</Button>
        </Stack>
        <LongText text={chunk.text} />
        {openedId === chunk.id && <Box sx={{ mt: 2 }}>
          {detail.isLoading ? <LoadingState label="Loading chunk metadata and embedding preview" /> : detail.isError ? <ErrorState error={detail.error} onRetry={() => void detail.refetch()} /> : detail.data && <Stack spacing={1.5}>
            <Typography variant="subtitle2">Metadata</Typography><JsonViewer value={detail.data.metadata} />
            <Typography variant="caption" color="text.secondary">Created {new Date(detail.data.created_at).toLocaleString()} · Vector ID: {detail.data.vector_id ?? 'None'}</Typography>
            {detail.data.embedding ? <Alert severity="info">Embedding: {detail.data.embedding.dimensions} dimensions · preview ({detail.data.embedding.preview.length} values): {detail.data.embedding.preview.join(', ')}{detail.data.embedding.truncated ? ' … (truncated)' : ''}</Alert> : <Alert severity="warning">Embedding preview unavailable. The vector may be absent or the vector store may be unreachable; this response does not distinguish those conditions.</Alert>}
          </Stack>}
        </Box>}
      </CardContent></Card>)}
      <Stack direction="row" gap={1} justifyContent="flex-end"><Button disabled={offset === 0 || page.isFetching} onClick={() => { setOffset(Math.max(0, offset - limit)); setOpenedId(undefined); }}>Previous</Button><Button disabled={page.isFetching || offset + limit >= (page.data?.total ?? 0)} onClick={() => { setOffset(offset + limit); setOpenedId(undefined); }}>Next</Button></Stack>
    </>}
  </Stack>;
}
