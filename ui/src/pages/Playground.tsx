import { useEffect, useMemo, useRef, useState } from 'react';
import { Alert, Box, Button, Card, CardContent, Chip, Collapse, Divider, FormControlLabel, Grid, IconButton, MenuItem, Select, Stack, Switch, TextField, Tooltip, Typography } from '@mui/material';
import PlayArrowRounded from '@mui/icons-material/PlayArrowRounded';
import StopRounded from '@mui/icons-material/StopRounded';
import DeleteSweepRounded from '@mui/icons-material/DeleteSweepRounded';
import EditRounded from '@mui/icons-material/EditRounded';
import ReplayRounded from '@mui/icons-material/ReplayRounded';
import TuneRounded from '@mui/icons-material/TuneRounded';
import { useSearchParams } from 'react-router-dom';
import { apiRequest } from '../api/client';
import { useModels } from '../api/hooks/useModels';
import { useDeployments } from '../api/hooks/useAdmin';
import { CopyButton, ErrorState, LoadingState, PageHeader, JsonViewer } from '../components/Shared';
import { streamChat, StreamHttpError } from '../features/playground/sse';
import { inspectorFromPlayground, mergeInspector, unavailableInspector } from '../features/playground/inspector';
import type { InspectorData, PlaygroundCollection, PlaygroundMessage, ReplayDetail, RerievalCitation, ToolCall } from '../features/playground/types';

interface ChatResponse { choices?: Array<{ message?: { content?: string | null; tool_calls?: ToolCall[] } }>; content?: string | null; tool_calls?: StreamToolCall[] }
interface StreamToolCall { index?: number; id?: string | null; name?: string | null; arguments?: string | null }
interface PlaygroundResponse { response: ChatResponse }
interface RagResponse { response: ChatResponse; sources: RerievalCitation[] }

const newId = (): string => `${Date.now()}-${Math.random().toString(36).slice(2)}`;
const parseList = (value: string): string[] => value.split(',').map((item) => item.trim()).filter(Boolean);
const safeJson = (value: unknown): string => JSON.stringify(value, null, 2) ?? 'null';
const parseArguments = (value: string): unknown => { try { return JSON.parse(value || '{}') as unknown; } catch { return value; } };

function Markdown({ value }: { value: string }) {
  const parts = value.split(/```([\w-]*)\n?([\s\S]*?)```/g);
  return <Stack spacing={1}>{parts.map((part, index) => index % 3 === 0 ? <Typography key={index} sx={{ whiteSpace: 'pre-wrap' }}>{part}</Typography> : index % 3 === 1 ? <Typography key={index} variant="caption" color="text.secondary">{part || 'code'}</Typography> : <Box key={index} sx={{ position: 'relative' }}><CopyButton value={part} label="Copy code" /><Box component="pre" sx={{ m: 0, p: 1.5, overflow: 'auto', borderRadius: 1, bgcolor: 'grey.900', color: 'grey.100', fontSize: 13 }}>{part}</Box></Box>)}</Stack>;
}

function Inspector({ data }: { data: InspectorData }) {
  return <Card variant="outlined"><CardContent><Typography variant="h3" sx={{ mb: 1.5 }}>Response inspector</Typography><Stack spacing={1}>{Object.entries(data).map(([key, value]) => <Stack key={key} direction="row" justifyContent="space-between" spacing={2}><Typography variant="body2" color="text.secondary">{key.replace(/[A-Z]/g, (letter) => ` ${letter}`).replace(/^./, (letter) => letter.toUpperCase())}</Typography><Typography variant="body2" sx={{ textAlign: 'right', fontFamily: 'monospace' }}>{value}</Typography></Stack>)}</Stack></CardContent></Card>;
}

function CitationList({ citations }: { citations: RerievalCitation[] }) {
  const [open, setOpen] = useState<string | null>(null);
  if (!citations.length) return null;
  return <Card variant="outlined"><CardContent><Typography variant="h3">Retrieved citations</Typography>{citations.map((citation) => <Box key={citation.id} sx={{ mt: 1.5 }}><Button fullWidth sx={{ justifyContent: 'space-between', textTransform: 'none' }} onClick={() => setOpen(open === citation.id ? null : citation.id)}>{citation.source ?? citation.document_id ?? 'Document'}<Chip size="small" label={citation.score.toFixed(3)} /></Button><Collapse in={open === citation.id}><Typography variant="body2" color="text.secondary" sx={{ px: 1.5, whiteSpace: 'pre-wrap' }}>{citation.text}</Typography></Collapse></Box>)}</CardContent></Card>;
}

export default function Playground() {
  const [searchParams] = useSearchParams();
  const models = useModels(100);
  const deployments = useDeployments(100);
  const [panelOpen, setPanelOpen] = useState(true);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [model, setModel] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [temperature, setTemperature] = useState('0.7');
  const [topP, setTopP] = useState('1');
  const [maxTokens, setMaxTokens] = useState('');
  const [stop, setStop] = useState('');
  const [streaming, setStreaming] = useState(true);
  const [noCache, setNoCache] = useState(false);
  const [routing, setRouting] = useState('');
  const [fallbacks, setFallbacks] = useState('');
  const [guardrail, setGuardrail] = useState('');
  const [tags, setTags] = useState('');
  const [ragMode, setRagMode] = useState(false);
  const [collection, setCollection] = useState('');
  const [collections, setCollections] = useState<PlaygroundCollection[]>([]);
  const [compareMode, setCompareMode] = useState(false);
  const [compareDeployments, setCompareDeployments] = useState<string[]>([]);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<PlaygroundMessage[]>([]);
  const [inspector, setInspector] = useState<InspectorData>(unavailableInspector);
  const [citations, setCitations] = useState<RerievalCitation[]>([]);
  const [error, setError] = useState<unknown>(null);
  const [running, setRunning] = useState(false);
  const abortRefs = useRef<Set<AbortController>>(new Set());

  useEffect(() => { const first = models.data?.items[0]?.name; if (!model && first) setModel(first); }, [models.data, model]);
  useEffect(() => { void apiRequest<PlaygroundCollection[]>('/v1/rag/collections').then(setCollections).catch(() => setCollections([])); }, []);
  useEffect(() => {
    const replay = searchParams.get('replay');
    if (!replay) return;
    void apiRequest<ReplayDetail>(`/admin/api/logs/${encodeURIComponent(replay)}`).then((detail) => {
      setModel(detail.model); setStreaming(detail.stream);
      const body = detail.request_body;
      const rawMessages = body?.messages;
      if (Array.isArray(rawMessages)) setMessages(rawMessages.flatMap((item) => typeof item === 'object' && item !== null && typeof (item as Record<string, unknown>).role === 'string' ? [{ id: newId(), role: (item as Record<string, string>).role as PlaygroundMessage['role'], content: typeof (item as Record<string, unknown>).content === 'string' ? (item as Record<string, string>).content : safeJson((item as Record<string, unknown>).content) }] : []));
      setError(body ? null : 'Replay restored the model and stream setting, but this log did not expose the original request body.');
    }).catch((replayError: unknown) => setError(replayError));
  }, [searchParams]);

  const availableDeployments = useMemo(() => deployments.data?.items.filter((item) => item.enabled) ?? [], [deployments.data]);
  const modelOptions = models.data?.items ?? [];
  const selectedDeployments = useMemo(() => compareDeployments.length ? compareDeployments : availableDeployments.filter((item) => item.model === model).slice(0, 1).map((item) => item.id), [availableDeployments, compareDeployments, model]);
  const buildBody = (deployment?: string): Record<string, unknown> => {
    const chatMessages = [...(systemPrompt ? [{ role: 'system', content: systemPrompt }] : []), ...messages.map(({ role, content, toolCalls, toolCallId }) => ({ role, content, ...(toolCalls ? { tool_calls: toolCalls } : {}), ...(toolCallId ? { tool_call_id: toolCallId } : {}) }))];
    return { model: deployment ?? model, messages: chatMessages, stream: streaming && !ragMode, ...(temperature ? { temperature: Number(temperature) } : {}), ...(topP ? { top_p: Number(topP) } : {}), ...(maxTokens ? { max_tokens: Number(maxTokens) } : {}), ...(stop ? { stop: parseList(stop) } : {}), no_cache: noCache, ...(routing ? { routing_strategy: routing } : {}), fallbacks: parseList(fallbacks), ...(guardrail ? { guardrail_policy: guardrail } : {}), tags: parseList(tags) };
  };
  const runOne = async (deployment: string, prompt: string): Promise<void> => {
    const controller = new AbortController(); abortRefs.current.add(controller);
    setError(null);
    const assistantId = newId(); setMessages((current) => [...current, { id: assistantId, role: 'assistant', content: '' }]);
    try {
      if (ragMode) {
        const body = { collection_id: collection, request: buildBody(deployment), query: prompt };
        const result = await apiRequest<RagResponse>('/v1/rag/query', { method: 'POST', body, signal: controller.signal });
        const content = result.response.choices?.[0]?.message?.content ?? '';
        setMessages((current) => current.map((item) => item.id === assistantId ? { ...item, content } : item)); setCitations(result.sources); setInspector(inspectorFromPlayground(result)); return;
      }
      const body = buildBody(deployment);
      if (body.stream !== true) {
        const result = await apiRequest<PlaygroundResponse>('/admin/api/playground/chat', { method: 'POST', body, signal: controller.signal });
        const message = result.response.choices?.[0]?.message;
        setMessages((current) => current.map((item) => item.id === assistantId ? { ...item, content: message?.content ?? '', toolCalls: message?.tool_calls } : item));
        setInspector(inspectorFromPlayground(result)); return;
      }
      const toolCalls = new Map<number, ToolCall>();
      let metadataReceived = false;
      const result = await streamChat('/admin/api/playground/chat', body, (event) => {
        if (event.event === 'metadata') {
          metadataReceived = true;
          setInspector((current) => mergeInspector(current, inspectorFromPlayground(event.data)));
          return;
        }
        if (event.event !== 'message' || typeof event.data !== 'object' || event.data === null) return;
        const payload = event.data as ChatResponse;
        const delta = typeof payload.content === 'string' ? payload.content : '';
        for (const call of payload.tool_calls ?? []) {
          const index = call.index ?? 0;
          const previous = toolCalls.get(index) ?? { id: call.id ?? `tool-${index}`, name: '', arguments: '' };
          toolCalls.set(index, { ...previous, id: call.id ?? previous.id, name: call.name ?? previous.name, arguments: previous.arguments + (call.arguments ?? '') });
        }
        if (delta || payload.tool_calls?.length) setMessages((current) => current.map((item) => item.id === assistantId ? { ...item, content: item.content + delta, toolCalls: toolCalls.size ? [...toolCalls.values()] : item.toolCalls } : item));
      }, controller.signal);
      if (!metadataReceived) setInspector(inspectorFromPlayground({}, result.headers));
      else setInspector((current) => mergeInspector(inspectorFromPlayground({}, result.headers), current));
    } catch (runError: unknown) { if (!(runError instanceof DOMException && runError.name === 'AbortError')) setError(runError); } finally { abortRefs.current.delete(controller); if (abortRefs.current.size === 0) setRunning(false); }
  };
  const send = () => { const prompt = input.trim(); if (!prompt || running) return; setRunning(true); setInput(''); setMessages((current) => [...current, { id: newId(), role: 'user', content: prompt }]); void Promise.all(selectedDeployments.map((deployment) => runOne(deployment, prompt))); };
  const stopRequest = () => { for (const controller of abortRefs.current) controller.abort(); setRunning(false); };
  const edit = (message: PlaygroundMessage) => { setInput(message.content); setMessages((current) => current.filter((item) => item.id !== message.id)); };
  const regenerate = (message: PlaygroundMessage) => { const previous = [...messages].reverse().find((item) => item.role === 'user'); if (previous) { setMessages((current) => current.filter((item) => item.id !== message.id)); void runOne(selectedDeployments[0] ?? model, previous.content); } };

  if (models.isLoading || deployments.isLoading) return <LoadingState label="Loading models and deployments" />;
  if (models.isError || deployments.isError) return <ErrorState error={models.error ?? deployments.error} />;
  const errorText = error instanceof StreamHttpError ? safeJson(error.payload) : error instanceof Error ? error.message : error == null ? '' : String(error);
  return <Box><PageHeader title="Playground" description="Exercise routing, caching, guardrails, RAG, and provider fallbacks from one request." action={<Stack direction="row" spacing={1}><Button size="small" onClick={() => setPanelOpen((value) => !value)} startIcon={<TuneRounded />}>{panelOpen ? 'Hide parameters' : 'Show parameters'}</Button><Button size="small" onClick={() => setInspectorOpen((value) => !value)}> {inspectorOpen ? 'Hide inspector' : 'Show inspector'}</Button></Stack>} />
    {error != null && <Alert severity="error" sx={{ mb: 2 }} action={<Button color="inherit" onClick={() => setError(null)}>Dismiss</Button>}><Typography sx={{ whiteSpace: 'pre-wrap' }}>{errorText}</Typography></Alert>}
    <Grid container spacing={2} alignItems="flex-start"><Grid item xs={12} md={panelOpen ? 3 : 0} sx={{ display: panelOpen ? 'block' : 'none' }}><Card><CardContent><Typography variant="h3">Parameters</Typography><Typography variant="caption" color="text.secondary">Gateway controls are marked separately from model parameters.</Typography><Stack spacing={1.5} sx={{ mt: 2 }}><Select size="small" value={model} onChange={(event) => setModel(event.target.value)} displayEmpty>{modelOptions.map((item) => <MenuItem key={item.name} value={item.name}>{item.name}</MenuItem>)}</Select><TextField label="System prompt" multiline minRows={2} value={systemPrompt} onChange={(event) => setSystemPrompt(event.target.value)} /><Stack direction="row" spacing={1}><TextField fullWidth label="Temperature" type="number" value={temperature} onChange={(event) => setTemperature(event.target.value)} /><TextField fullWidth label="Top P" type="number" value={topP} onChange={(event) => setTopP(event.target.value)} /></Stack><TextField label="Max tokens" type="number" value={maxTokens} onChange={(event) => setMaxTokens(event.target.value)} /><TextField label="Stop sequences (comma separated)" value={stop} onChange={(event) => setStop(event.target.value)} /><FormControlLabel control={<Switch checked={streaming} onChange={(event) => setStreaming(event.target.checked)} />} label="Stream response" /><Divider /><Typography variant="caption" color="primary">Gateway controls</Typography><FormControlLabel control={<Switch checked={noCache} onChange={(event) => setNoCache(event.target.checked)} />} label="Bypass cache (no_cache)" /><TextField label="Routing strategy override" value={routing} onChange={(event) => setRouting(event.target.value)} /><TextField label="Explicit fallback chain" placeholder="deployment-a, deployment-b" value={fallbacks} onChange={(event) => setFallbacks(event.target.value)} /><TextField label="Guardrail policy" value={guardrail} onChange={(event) => setGuardrail(event.target.value)} /><TextField label="Tags" placeholder="demo, cost-test" value={tags} onChange={(event) => setTags(event.target.value)} /><Divider /><FormControlLabel control={<Switch checked={ragMode} onChange={(event) => setRagMode(event.target.checked)} />} label="RAG-grounded mode" />{ragMode && <Select size="small" value={collection} onChange={(event) => setCollection(event.target.value)} displayEmpty><MenuItem value="">Select collection</MenuItem>{collections.map((item) => <MenuItem key={item.id} value={item.id}>{item.name} ({item.document_count} docs)</MenuItem>)}</Select>}<FormControlLabel control={<Switch checked={compareMode} onChange={(event) => setCompareMode(event.target.checked)} />} label="Compare deployments" />{compareMode && <Select multiple size="small" value={compareDeployments} onChange={(event) => setCompareDeployments(typeof event.target.value === 'string' ? event.target.value.split(',') : (event.target.value as string[]))}>{availableDeployments.map((item) => <MenuItem key={item.id} value={item.id}>{item.id} · {item.provider}</MenuItem>)}</Select>}</Stack></CardContent></Card></Grid>
      <Grid item xs={12} md={panelOpen && inspectorOpen ? 6 : panelOpen || inspectorOpen ? 9 : 12}><Card sx={{ minHeight: 560, display: 'flex', flexDirection: 'column' }}><CardContent sx={{ flex: 1, overflow: 'auto' }}><Stack spacing={2}>{messages.length === 0 && <Typography color="text.secondary" sx={{ py: 10, textAlign: 'center' }}>Send a message to begin. Every response is measured by the gateway.</Typography>}{messages.map((message) => <Box key={message.id} sx={{ alignSelf: message.role === 'user' ? 'flex-end' : 'stretch', maxWidth: message.role === 'user' ? '85%' : '100%' }}><Stack direction="row" spacing={1} alignItems="center"><Chip size="small" label={message.role} color={message.role === 'assistant' ? 'primary' : 'default'} />{message.role === 'user' && <Tooltip title="Edit and resend"><IconButton size="small" onClick={() => edit(message)}><EditRounded fontSize="small" /></IconButton></Tooltip>}{message.role === 'assistant' && <Tooltip title="Regenerate"><IconButton size="small" onClick={() => regenerate(message)}><ReplayRounded fontSize="small" /></IconButton></Tooltip>}</Stack><Card variant="outlined" sx={{ mt: 0.5, p: 1.5, bgcolor: message.role === 'user' ? 'action.hover' : 'background.paper' }}>{message.role === 'assistant' ? <Markdown value={message.content} /> : <Typography sx={{ whiteSpace: 'pre-wrap' }}>{message.content}</Typography>}{running && message.role === 'assistant' && message.id === messages[messages.length - 1]?.id && <Typography component="span" sx={{ ml: 0.5, animation: 'blink 1s step-end infinite' }}>▍</Typography>}</Card>{message.toolCalls?.map((tool) => <Card key={tool.id} variant="outlined" sx={{ mt: 1, borderColor: 'warning.main' }}><CardContent><Typography variant="subtitle2">Tool call: {tool.name}</Typography><JsonViewer value={parseArguments(tool.arguments)} /><TextField fullWidth label="Tool result" multiline minRows={2} value={tool.result ?? ''} onChange={(event) => setMessages((current) => current.map((item) => item.id === message.id ? { ...item, toolCalls: item.toolCalls?.map((call) => call.id === tool.id ? { ...call, result: event.target.value } : call) } : item))} /></CardContent></Card>)}</Box>)}</Stack></CardContent><Divider /><Stack direction="row" spacing={1} sx={{ p: 1.5 }}><TextField fullWidth multiline maxRows={4} placeholder="Ask the gateway…" value={input} onChange={(event) => setInput(event.target.value)} onKeyDown={(event) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); send(); } }} /><Button variant="contained" onClick={running ? stopRequest : send} color={running ? 'error' : 'primary'} startIcon={running ? <StopRounded /> : <PlayArrowRounded />}>{running ? 'Stop' : 'Send'}</Button><Button onClick={() => setMessages([])} startIcon={<DeleteSweepRounded />}>Clear</Button></Stack></Card></Grid>
      {inspectorOpen && <Grid item xs={12} md={3}><Inspector data={inspector} />{citations.length > 0 && <Box sx={{ mt: 2 }}><CitationList citations={citations} /></Box>}</Grid>}</Grid></Box>;
}
