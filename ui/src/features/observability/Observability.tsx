import { useTheme } from '@mui/material/styles';
import { Alert, Box, Card, CardContent, FormControl, InputLabel, MenuItem, Paper, Select, Stack, Typography } from '@mui/material';
import type { ReactNode } from 'react';
import { EmptyState } from '../../components/Shared';

export const WINDOWS = ['1h', '24h', '7d', '30d'] as const;
export type WindowValue = typeof WINDOWS[number];

export function WindowSelect({ value, onChange }: { value: WindowValue; onChange: (value: WindowValue) => void }) {
  return <FormControl size="small" sx={{ minWidth: 145 }}>
    <InputLabel id="observability-window-label">Time window</InputLabel>
    <Select labelId="observability-window-label" value={value} label="Time window" onChange={(event) => onChange(event.target.value as WindowValue)}>
      <MenuItem value="1h">Last hour</MenuItem><MenuItem value="24h">Last 24 hours</MenuItem>
      <MenuItem value="7d">Last 7 days</MenuItem><MenuItem value="30d">Last 30 days</MenuItem>
    </Select>
  </FormControl>;
}

export function ChartPanel({ title, description, children }: { title: string; description?: string; children: ReactNode }) {
  return <Paper sx={{ p: 2.5, minWidth: 0 }}><Typography variant="h3">{title}</Typography>
    {description && <Typography variant="body2" color="text.secondary" sx={{ mt: 0.5 }}>{description}</Typography>}
    <Box sx={{ mt: 2, height: 280, minWidth: 0 }}>{children}</Box>
  </Paper>;
}

export function UnavailableChart({ message }: { message: string }) {
  return <EmptyState title="Not available" description={message} />;
}

export function useChartColors() {
  const theme = useTheme();
  return { primary: theme.palette.primary.main, secondary: theme.palette.secondary.main, success: theme.palette.success.main,
    error: theme.palette.error.main, warning: theme.palette.warning.main, text: theme.palette.text.secondary,
    grid: theme.palette.divider, paper: theme.palette.background.paper };
}

export function UnavailableNote({ children }: { children: ReactNode }) {
  return <Alert severity="info" variant="outlined" sx={{ mt: 1 }}>{children}</Alert>;
}

export function SectionHeading({ title, subtitle }: { title: string; subtitle?: string }) {
  return <Stack spacing={0.5} sx={{ mb: 1.5 }}><Typography variant="h3">{title}</Typography>
    {subtitle && <Typography variant="body2" color="text.secondary">{subtitle}</Typography>}</Stack>;
}

export function MetricTrend({ current, previous, unit, favorable, format }: { current: number; previous: number; unit?: string; favorable: 'up' | 'down' | 'neutral'; format: (value: number) => string }) {
  const change = current - previous;
  const direction = change > 0 ? 'up' : change < 0 ? 'down' : 'flat';
  const beneficial = (direction === 'up' && favorable === 'up') || (direction === 'down' && favorable === 'down');
  const color = direction === 'flat' || favorable === 'neutral' ? 'text.secondary' : beneficial ? 'success.main' : 'error.main';
  const magnitude = unit === 'percentage points' ? `${(Math.abs(change) * 100).toFixed(1)} pp` : `${format(Math.abs(change))}${unit ? ` ${unit}` : ''}`;
  return <Typography variant="caption" color={color} sx={{ display: 'block', mt: 0.5 }}>
    {direction === 'up' ? '↑' : direction === 'down' ? '↓' : '→'} {magnitude} vs previous window
  </Typography>;
}

export function MetricStatCard({ label, value, hint, current, previous, favorable, unit, format }: {
  label: string; value: string; hint: string; current: number; previous: number; favorable: 'up' | 'down' | 'neutral'; unit?: string; format: (value: number) => string;
}) {
  return <Card sx={{ height: '100%' }}><CardContent sx={{ p: 2.5, '&:last-child': { pb: 2.5 } }}>
    <Typography variant="body2" color="text.secondary">{label}</Typography>
    <Typography variant="h2" sx={{ mt: 2, fontVariantNumeric: 'tabular-nums' }}>{value}</Typography>
    <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>{hint}</Typography>
    <MetricTrend current={current} previous={previous} favorable={favorable} unit={unit} format={format} />
  </CardContent></Card>;
}
